# %%
"""
Fit condition-by-condition with 3 free parameters:
gamma, omega, t_E_aff.

For each animal, w and del_go are fixed to posterior means from the
animal-wise NPL + alpha + ABL-specific ILD2-delay fit:

    fit_animal_by_animal/NPL_alpha_ABL_specific_ILD2_delay_fit_results/
    results_<batch>_animal_<animal>_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl

Abort parameters are loaded from the usual abort-source pickle folder.
Pickles and corner plots are saved in separate folders.
"""

import datetime as dt
import os
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyvbmc import VBMC
import pyvbmc.vbmc.active_sample as pyvbmc_active_sample
import pyvbmc.vbmc.variational_optimization as pyvbmc_variational_optimization

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table
from led_off_gamma_omega_pdf_utils import (
    cum_pro_and_reactive_trunc_fn,
    up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn,
)


# %%
# =============================================================================
# PyVBMC compatibility patch
# =============================================================================
# PyVBMC can return one-element arrays for full-ELBO scalar variance terms with
# some GP states. Newer NumPy refuses assigning those arrays into scalar slots,
# so coerce only the scalar diagnostics while leaving gradients and per-component
# arrays unchanged.
if not getattr(pyvbmc_variational_optimization, "_ddm_scalar_elcbo_patch", False):
    _original_neg_elcbo = pyvbmc_variational_optimization._neg_elcbo

    def _scalar_like(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(np.mean(arr))

    def _neg_elcbo_with_scalar_diagnostics(*args, **kwargs):
        result = _original_neg_elcbo(*args, **kwargs)
        if not isinstance(result, tuple):
            return result

        result = list(result)
        if len(result) == 11:
            for idx in [0, 2, 3, 4, 6, 7, 8]:
                result[idx] = _scalar_like(result[idx])
        elif len(result) == 5:
            for idx in [0, 2, 3, 4]:
                result[idx] = _scalar_like(result[idx])
        return tuple(result)

    pyvbmc_variational_optimization._neg_elcbo = _neg_elcbo_with_scalar_diagnostics
    pyvbmc_active_sample._neg_elcbo = _neg_elcbo_with_scalar_diagnostics
    pyvbmc_variational_optimization._ddm_scalar_elcbo_patch = True


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
desired_batches_override = os.environ.get("DESIRED_BATCHES_OVERRIDE")
if desired_batches_override:
    DESIRED_BATCHES = [
        batch.strip() for batch in desired_batches_override.split(",") if batch.strip()
    ]

default_batch_dir = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
if not any(
    os.path.exists(os.path.join(default_batch_dir, f"batch_{batch}_valid_and_aborts.csv"))
    for batch in DESIRED_BATCHES
):
    raw_batch_dir = os.path.join(REPO_DIR, "raw_data", "batch_csvs")
    if any(
        os.path.exists(os.path.join(raw_batch_dir, f"batch_{batch}_valid_and_aborts.csv"))
        for batch in DESIRED_BATCHES
    ):
        default_batch_dir = raw_batch_dir

batch_dir = os.environ.get("BATCH_CSV_DIR_OVERRIDE", default_batch_dir)
abort_params_dir = os.environ.get(
    "ABORT_PARAMS_DIR_OVERRIDE",
    os.path.join(REPO_DIR, "aborts_ipl_npl_time_fit_results"),
)
abl_specific_result_dir = os.environ.get(
    "ABL_SPECIFIC_RESULT_DIR_OVERRIDE",
    os.path.join(REPO_DIR, "fit_animal_by_animal", "NPL_alpha_ABL_specific_ILD2_delay_fit_results"),
)

OUTPUT_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files",
)
CORNER_PLOT_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots",
)
FAILURE_MARKER_FOLDER = os.path.join(
    SCRIPT_DIR,
    "each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_failed_conditions",
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CORNER_PLOT_FOLDER, exist_ok=True)
os.makedirs(FAILURE_MARKER_FOLDER, exist_ok=True)

ABL_SPECIFIC_RESULT_KEY = "vbmc_norm_alpha_abl_specific_ild2_delay_tied_results"
FILENAME_SUFFIX = "_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params"

all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
K_max = 10
N_JOBS = int(os.environ.get("N_JOBS", "30"))
N_POSTERIOR_SAMPLES_FOR_CORNER = int(float(os.environ.get("N_POSTERIOR_SAMPLES_FOR_CORNER", "5e4")))
OVERWRITE_FITS = False
DRY_RUN = False
MAX_ANIMALS = None
MAX_CONDITIONS = None
BATCH_ANIMAL_PAIRS_OVERRIDE = os.environ.get("BATCH_ANIMAL_PAIRS_OVERRIDE")

# Environment overrides are for smoke tests and watcher-style one-animal runs.
DRY_RUN = os.environ.get("DRY_RUN", str(int(DRY_RUN))).lower() in {"1", "true", "yes", "y"}
OVERWRITE_FITS = os.environ.get("OVERWRITE_FITS", str(int(OVERWRITE_FITS))).lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MAX_ANIMALS_ENV = os.environ.get("MAX_ANIMALS")
MAX_CONDITIONS_ENV = os.environ.get("MAX_CONDITIONS")
if MAX_ANIMALS_ENV not in [None, ""]:
    MAX_ANIMALS = int(MAX_ANIMALS_ENV)
if MAX_CONDITIONS_ENV not in [None, ""]:
    MAX_CONDITIONS = int(MAX_CONDITIONS_ENV)


# %%
# =============================================================================
# Load animal list
# =============================================================================
if BATCH_ANIMAL_PAIRS_OVERRIDE:
    batch_animal_pairs = []
    for pair_text in BATCH_ANIMAL_PAIRS_OVERRIDE.split(","):
        pair_text = pair_text.strip()
        if not pair_text:
            continue
        batch_name, animal_id = pair_text.replace("/", ":").split(":")
        batch_animal_pairs.append((batch_name.strip(), int(animal_id)))
else:
    batch_animal_pairs = load_batch_animal_pairs(batch_dir, DESIRED_BATCHES)

if MAX_ANIMALS is not None:
    batch_animal_pairs = batch_animal_pairs[:MAX_ANIMALS]
print_batch_animal_table(batch_animal_pairs)

conditions_to_fit = [(ABL, ILD) for ABL in all_ABLs_cond for ILD in all_ILDs_cond]
if MAX_CONDITIONS is not None:
    conditions_to_fit = conditions_to_fit[:MAX_CONDITIONS]

print(f"Batch CSV dir: {batch_dir}")
print(f"Abort params dir: {abort_params_dir}")
print(f"ABL-specific result dir: {abl_specific_result_dir}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Corner plot folder: {CORNER_PLOT_FOLDER}")
print(f"Failure marker folder: {FAILURE_MARKER_FOLDER}")
print(f"Fitting {len(batch_animal_pairs)} animals x {len(conditions_to_fit)} conditions")
print(
    f"DRY_RUN={DRY_RUN}, OVERWRITE_FITS={OVERWRITE_FITS}, "
    f"N_JOBS={N_JOBS}, N_POSTERIOR_SAMPLES_FOR_CORNER={N_POSTERIOR_SAMPLES_FOR_CORNER}"
)


# %%
# =============================================================================
# Helper functions
# =============================================================================
def get_abort_params_from_animal_pkl_file(batch_name, animal_id):
    """Load abort parameters (V_A, theta_A, t_A_aff) from abort-source pkl file."""
    pkl_file = os.path.join(abort_params_dir, f"results_{batch_name}_animal_{animal_id}.pkl")
    if not os.path.exists(pkl_file):
        return None

    with open(pkl_file, "rb") as f:
        fit_results_data = pickle.load(f)

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname not in fit_results_data:
        return None

    abort_samples = fit_results_data[abort_keyname]
    return {
        "V_A": float(np.mean(abort_samples["V_A_samples"])),
        "theta_A": float(np.mean(abort_samples["theta_A_samples"])),
        "t_A_aff": float(np.mean(abort_samples["t_A_aff_samp"])),
    }


def failure_marker_file_for_condition(batch_name, animal_id, cond_ABL, cond_ILD):
    return os.path.join(
        FAILURE_MARKER_FOLDER,
        f"failed_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.log",
    )


def record_condition_failure(batch_name, animal_id, cond_ABL, cond_ILD, n_trials, error, stage):
    marker_file = failure_marker_file_for_condition(batch_name, animal_id, cond_ABL, cond_ILD)
    with open(marker_file, "a", encoding="utf-8") as f:
        f.write(f"attempt_at={dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"batch={batch_name}\n")
        f.write(f"animal={animal_id}\n")
        f.write(f"ABL={cond_ABL}\n")
        f.write(f"ILD={cond_ILD}\n")
        f.write(f"n_trials={n_trials}\n")
        f.write(f"stage={stage}\n")
        f.write(f"error_type={type(error).__name__}\n")
        f.write(f"error_message={error}\n\n")
    return marker_file


def clear_condition_failure_marker(batch_name, animal_id, cond_ABL, cond_ILD):
    marker_file = failure_marker_file_for_condition(batch_name, animal_id, cond_ABL, cond_ILD)
    if os.path.exists(marker_file):
        os.remove(marker_file)


def get_fixed_w_del_go_from_abl_specific_result(batch_name, animal_id):
    result_pkl = os.path.join(
        abl_specific_result_dir,
        f"results_{batch_name}_animal_{animal_id}_NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS.pkl",
    )
    if not os.path.exists(result_pkl):
        return None

    with open(result_pkl, "rb") as f:
        fit_results_data = pickle.load(f)

    if ABL_SPECIFIC_RESULT_KEY not in fit_results_data:
        print(f"  WARNING: {ABL_SPECIFIC_RESULT_KEY} missing in {result_pkl}")
        return None

    result_samples = fit_results_data[ABL_SPECIFIC_RESULT_KEY]
    required_keys = ["w_samples", "del_go_samples"]
    missing_keys = [key for key in required_keys if key not in result_samples]
    if missing_keys:
        print(f"  WARNING: Missing {missing_keys} in {result_pkl}")
        return None

    vbmc_message = str(result_samples.get("message", ""))
    if "stable" not in vbmc_message.lower():
        short_message = vbmc_message.replace("\n", " ")[:120] or "missing VBMC message"
        print(f"  WARNING: ABL-specific upstream fit is not stable yet: {short_message}")
        return None

    return {
        "w": float(np.mean(result_samples["w_samples"])),
        "del_go": float(np.mean(result_samples["del_go_samples"])),
        "result_pkl": result_pkl,
        "elbo": result_samples.get("elbo", np.nan),
        "message": result_samples.get("message", ""),
    }


def trapezoidal_logpdf(x, a, b, c, d):
    """Trapezoidal prior log-pdf."""
    if x < a or x > d:
        return -np.inf

    area = ((b - a) + (d - c)) / 2 + (c - b)
    h_max = 1.0 / area

    if a <= x <= b:
        pdf_value = ((x - a) / (b - a)) * h_max
    elif b < x < c:
        pdf_value = h_max
    elif c <= x <= d:
        pdf_value = ((d - x) / (d - c)) * h_max
    else:
        pdf_value = 0.0

    if pdf_value <= 0.0:
        return -np.inf
    return np.log(pdf_value)


def save_corner_plot(vbmc, corner_plot_file, fixed_w, fixed_del_go, batch_name, animal_id, cond_ABL, cond_ILD):
    samples = vbmc.vp.sample(N_POSTERIOR_SAMPLES_FOR_CORNER)[0]
    fig = corner.corner(
        samples,
        labels=["gamma", "omega", "t_E_aff"],
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.025, 0.5, 0.975],
    )
    fig.suptitle(
        (
            f"{batch_name}/{animal_id}, ABL={cond_ABL}, ILD={cond_ILD}, "
            f"fixed w={fixed_w:.4f}, fixed del_go={fixed_del_go:.4f}"
        ),
        fontsize=12,
    )
    fig.savefig(corner_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
# =============================================================================
# Parameter bounds (3 params: gamma, omega, t_E_aff)
# =============================================================================
omega_bounds = [0.1, 15]
omega_plausible_bounds = [2, 12]

t_E_aff_bounds = [0, 1]
t_E_aff_plausible_bounds = [0.01, 0.2]


# %%
# =============================================================================
# Main fitting loop
# =============================================================================
total_completed = 0
total_existing = 0
total_corners_generated = 0
total_low_trial_skips = 0
total_missing_abl_specific_results = 0
total_errors = 0

for batch_name, animal_id in batch_animal_pairs:
    print("\n" + "=" * 70)
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print("=" * 70)

    fixed_params = get_fixed_w_del_go_from_abl_specific_result(batch_name, animal_id)
    if fixed_params is None:
        print("  ABL-specific fit result not found or incomplete. Skipping animal.")
        total_missing_abl_specific_results += 1
        continue

    fixed_w = fixed_params["w"]
    fixed_del_go = fixed_params["del_go"]

    abort_params = get_abort_params_from_animal_pkl_file(batch_name, animal_id)
    if abort_params is None:
        print(f"  WARNING: No abort params found for {batch_name}/{animal_id}. Skipping animal.")
        total_errors += 1
        continue

    V_A = abort_params["V_A"]
    theta_A = abort_params["theta_A"]
    t_A_aff = abort_params["t_A_aff"]

    batch_file = os.path.join(batch_dir, f"batch_{batch_name}_valid_and_aborts.csv")
    if not os.path.exists(batch_file):
        print(f"  WARNING: Batch CSV not found: {batch_file}. Skipping animal.")
        total_errors += 1
        continue

    df = pd.read_csv(batch_file)
    df_animal = df[df["animal"] == int(animal_id)]
    df_animal_success = df_animal[df_animal["success"].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[
        (df_animal_success["RTwrtStim"] <= 1) & (df_animal_success["RTwrtStim"] > 0)
    ]

    print(f"  fixed w from ABL-specific fit: {fixed_w:.6f}")
    print(f"  fixed del_go from ABL-specific fit: {fixed_del_go:.6f}")
    print(f"  ABL-specific source: {fixed_params['result_pkl']}")
    print(f"  abort params: V_A={V_A:.6f}, theta_A={theta_A:.6f}, t_A_aff={t_A_aff:.6f}")
    print(f"  valid RT-filtered trials: {len(df_animal_success_rt_filter)}")

    animal_completed = 0
    animal_existing = 0
    animal_corners_generated = 0
    animal_low_trial_skips = 0
    animal_errors = 0

    for cond_ABL, cond_ILD in conditions_to_fit:
        pkl_file = os.path.join(
            OUTPUT_FOLDER,
            f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.pkl",
        )
        corner_plot_file = os.path.join(
            CORNER_PLOT_FOLDER,
            f"corner_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}{FILENAME_SUFFIX}.png",
        )

        df_cond = df_animal_success_rt_filter[
            (df_animal_success_rt_filter["ABL"] == cond_ABL)
            & (df_animal_success_rt_filter["ILD"] == cond_ILD)
        ]
        n_trials = len(df_cond)

        if n_trials < 10:
            print(f"  [{cond_ABL}, {cond_ILD:+3d}] Only {n_trials} trials, skipping")
            animal_low_trial_skips += 1
            total_low_trial_skips += 1
            continue

        if DRY_RUN:
            print(
                f"  [{cond_ABL}, {cond_ILD:+3d}] DRY_RUN: {n_trials} trials, "
                f"pickle -> {pkl_file}, corner -> {corner_plot_file}"
            )
            continue

        if os.path.exists(pkl_file) and not OVERWRITE_FITS:
            print(f"  [{cond_ABL}, {cond_ILD:+3d}] Pickle exists, skipping fit")
            animal_existing += 1
            total_existing += 1

            if not os.path.exists(corner_plot_file):
                try:
                    with open(pkl_file, "rb") as f:
                        existing_vbmc = pickle.load(f)
                    save_corner_plot(
                        existing_vbmc,
                        corner_plot_file,
                        fixed_w,
                        fixed_del_go,
                        batch_name,
                        animal_id,
                        cond_ABL,
                        cond_ILD,
                    )
                    print(f"    -> Saved missing corner: {os.path.basename(corner_plot_file)}")
                    animal_corners_generated += 1
                    total_corners_generated += 1
                except Exception as e:
                    print(f"    -> ERROR while saving missing corner: {e}")
                    marker_file = record_condition_failure(
                        batch_name,
                        animal_id,
                        cond_ABL,
                        cond_ILD,
                        n_trials,
                        e,
                        "corner_from_existing_pickle",
                    )
                    print(f"    -> Recorded failure marker: {marker_file}")
                    animal_errors += 1
                    total_errors += 1
                else:
                    clear_condition_failure_marker(batch_name, animal_id, cond_ABL, cond_ILD)
            continue

        print(f"  [{cond_ABL}, {cond_ILD:+3d}] Fitting with {n_trials} trials...")

        def compute_loglike_trial(row, gamma, omega, t_E_aff):
            c_A_trunc_time = 0.3
            if "timed_fix" in row:
                rt = row["timed_fix"]
            else:
                rt = row["TotalFixTime"]
            t_stim = row["intended_fix"]
            response_poke = row["response_poke"]

            trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                t_stim + 1,
                c_A_trunc_time,
                V_A,
                theta_A,
                t_A_aff,
                t_stim,
                t_E_aff,
                gamma,
                omega,
                fixed_w,
                K_max,
            ) - cum_pro_and_reactive_trunc_fn(
                t_stim,
                c_A_trunc_time,
                V_A,
                theta_A,
                t_A_aff,
                t_stim,
                t_E_aff,
                gamma,
                omega,
                fixed_w,
                K_max,
            )

            choice = 2 * response_poke - 5
            P_joint_rt_choice = up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn(
                rt,
                V_A,
                theta_A,
                gamma,
                omega,
                t_stim,
                t_A_aff,
                t_E_aff,
                fixed_del_go,
                choice,
                fixed_w,
                K_max,
            )

            P_joint_rt_choice_trunc = max(
                P_joint_rt_choice / (trunc_factor_p_joint + 1e-10),
                1e-10,
            )
            return np.log(P_joint_rt_choice_trunc)

        if cond_ILD > 0:
            gamma_bounds = [-1, 5]
            gamma_plausible_bounds = [0, 3]
        else:
            gamma_bounds = [-5, 1]
            gamma_plausible_bounds = [-3, 0]

        def vbmc_prior_fn(params):
            gamma, omega, t_E_aff = params
            gamma_logpdf = trapezoidal_logpdf(
                gamma,
                gamma_bounds[0],
                gamma_plausible_bounds[0],
                gamma_plausible_bounds[1],
                gamma_bounds[1],
            )
            omega_logpdf = trapezoidal_logpdf(
                omega,
                omega_bounds[0],
                omega_plausible_bounds[0],
                omega_plausible_bounds[1],
                omega_bounds[1],
            )
            t_E_aff_logpdf = trapezoidal_logpdf(
                t_E_aff,
                t_E_aff_bounds[0],
                t_E_aff_plausible_bounds[0],
                t_E_aff_plausible_bounds[1],
                t_E_aff_bounds[1],
            )
            return float(gamma_logpdf + omega_logpdf + t_E_aff_logpdf)

        def vbmc_loglike_fn(params):
            gamma, omega, t_E_aff = params
            all_loglike = Parallel(n_jobs=N_JOBS)(
                delayed(compute_loglike_trial)(row, gamma, omega, t_E_aff)
                for _, row in df_cond.iterrows()
            )
            return float(np.sum(all_loglike))

        def vbmc_joint_fn(params):
            return float(vbmc_prior_fn(params) + vbmc_loglike_fn(params))

        lb = np.array([gamma_bounds[0], omega_bounds[0], t_E_aff_bounds[0]])
        ub = np.array([gamma_bounds[1], omega_bounds[1], t_E_aff_bounds[1]])
        plb = np.array(
            [
                gamma_plausible_bounds[0],
                omega_plausible_bounds[0],
                t_E_aff_plausible_bounds[0],
            ]
        )
        pub = np.array(
            [
                gamma_plausible_bounds[1],
                omega_plausible_bounds[1],
                t_E_aff_plausible_bounds[1],
            ]
        )

        np.random.seed(42)
        x_0 = np.array(
            [
                np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1]),
                np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1]),
                np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1]),
            ]
        )

        try:
            vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={"display": "iter"})
            vp, results = vbmc.optimize()

            vbmc.save(pkl_file, overwrite=True)
            print(f"    -> Saved pickle: {os.path.basename(pkl_file)}")
            animal_completed += 1
            total_completed += 1

            save_corner_plot(
                vbmc,
                corner_plot_file,
                fixed_w,
                fixed_del_go,
                batch_name,
                animal_id,
                cond_ABL,
                cond_ILD,
            )
            print(f"    -> Saved corner: {os.path.basename(corner_plot_file)}")
            animal_corners_generated += 1
            total_corners_generated += 1
            clear_condition_failure_marker(batch_name, animal_id, cond_ABL, cond_ILD)
        except Exception as e:
            print(f"    -> ERROR: {e}")
            marker_file = record_condition_failure(
                batch_name,
                animal_id,
                cond_ABL,
                cond_ILD,
                n_trials,
                e,
                "vbmc_fit_or_save",
            )
            print(f"    -> Recorded failure marker: {marker_file}")
            animal_errors += 1
            total_errors += 1

    print(
        "  Animal summary: "
        f"completed={animal_completed}, existing={animal_existing}, "
        f"corners_generated={animal_corners_generated}, "
        f"low_trial_skips={animal_low_trial_skips}, errors={animal_errors}"
    )

print("\n" + "=" * 70)
print(
    "DONE: "
    f"completed={total_completed}, existing={total_existing}, "
    f"corners_generated={total_corners_generated}, "
    f"low_trial_skips={total_low_trial_skips}, "
    f"missing_abl_specific_results={total_missing_abl_specific_results}, "
    f"errors={total_errors}"
)
print("=" * 70)

# %%
