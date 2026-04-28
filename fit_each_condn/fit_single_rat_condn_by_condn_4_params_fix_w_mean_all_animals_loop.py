# %%
"""
Fit condition-by-condition with 4 free parameters:
gamma, omega, t_E_aff, del_go.

For each animal, w is fixed to the animal's mean_w from:
five_param_w_mean_median_by_animal.csv

This fit is condition-by-condition for all animals across all selected batches.
Pickles and corner plots are saved in separate folders.
"""

import os
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyvbmc import VBMC

from gamma_omega_alpha_utils import load_batch_animal_pairs, print_batch_animal_table
from led_off_gamma_omega_pdf_utils import (
    cum_pro_and_reactive_trunc_fn,
    up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn,
)


# %%
# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
batch_dir = os.path.join(REPO_DIR, "fit_animal_by_animal", "batch_csvs")
W_SUMMARY_CSV_PATH = os.path.join(SCRIPT_DIR, "five_param_w_mean_median_by_animal.csv")

OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_4_params_fix_w_mean_pkl_files")
CORNER_PLOT_FOLDER = os.path.join(SCRIPT_DIR, "each_animal_cond_fit_4_params_fix_w_mean_corner_plots")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CORNER_PLOT_FOLDER, exist_ok=True)

all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
K_max = 10
N_JOBS = 30
N_POSTERIOR_SAMPLES_FOR_CORNER = int(5e4)
OVERWRITE_FITS = False
DRY_RUN = False
MAX_ANIMALS = None
MAX_CONDITIONS = None

# Environment overrides are only for smoke tests; checked-in defaults above are the full fit.
DRY_RUN = os.environ.get("DRY_RUN", str(int(DRY_RUN))).lower() in {"1", "true", "yes", "y"}
MAX_ANIMALS_ENV = os.environ.get("MAX_ANIMALS")
MAX_CONDITIONS_ENV = os.environ.get("MAX_CONDITIONS")
if MAX_ANIMALS_ENV not in [None, ""]:
    MAX_ANIMALS = int(MAX_ANIMALS_ENV)
if MAX_CONDITIONS_ENV not in [None, ""]:
    MAX_CONDITIONS = int(MAX_CONDITIONS_ENV)


# %%
# =============================================================================
# Load animal list and fixed w values
# =============================================================================
batch_animal_pairs = load_batch_animal_pairs(batch_dir, DESIRED_BATCHES)
if MAX_ANIMALS is not None:
    batch_animal_pairs = batch_animal_pairs[:MAX_ANIMALS]
print_batch_animal_table(batch_animal_pairs)

w_summary_df = pd.read_csv(W_SUMMARY_CSV_PATH)
w_summary_df["animal"] = w_summary_df["animal"].astype(str)
w_by_batch_animal = {
    (row["batch_name"], row["animal"]): float(row["mean_w"])
    for _, row in w_summary_df.iterrows()
}
conditions_to_fit = [(ABL, ILD) for ABL in all_ABLs_cond for ILD in all_ILDs_cond]
if MAX_CONDITIONS is not None:
    conditions_to_fit = conditions_to_fit[:MAX_CONDITIONS]

print(f"Loaded fixed mean_w values for {len(w_by_batch_animal)} animals from {W_SUMMARY_CSV_PATH}")
print(f"Fitting {len(batch_animal_pairs)} animals x {len(conditions_to_fit)} conditions")
print(f"DRY_RUN={DRY_RUN}, OVERWRITE_FITS={OVERWRITE_FITS}")


# %%
# =============================================================================
# Helper functions
# =============================================================================
def get_abort_params_from_animal_pkl_file(batch_name, animal_id):
    """Load abort parameters (V_A, theta_A, t_A_aff) from animal pkl file."""
    pkl_file = os.path.join(
        REPO_DIR,
        "fit_animal_by_animal",
        f"results_{batch_name}_animal_{animal_id}.pkl",
    )
    if not os.path.exists(pkl_file):
        return None

    with open(pkl_file, "rb") as f:
        fit_results_data = pickle.load(f)

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname not in fit_results_data:
        return None

    abort_samples = fit_results_data[abort_keyname]
    return {
        "V_A": np.mean(abort_samples["V_A_samples"]),
        "theta_A": np.mean(abort_samples["theta_A_samples"]),
        "t_A_aff": np.mean(abort_samples["t_A_aff_samp"]),
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


def save_corner_plot(vbmc, corner_plot_file, fixed_w, batch_name, animal_id, cond_ABL, cond_ILD):
    samples = vbmc.vp.sample(N_POSTERIOR_SAMPLES_FOR_CORNER)[0]
    fig = corner.corner(
        samples,
        labels=["gamma", "omega", "t_E_aff", "del_go"],
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.025, 0.5, 0.975],
    )
    fig.suptitle(
        f"{batch_name}/{animal_id}, ABL={cond_ABL}, ILD={cond_ILD}, fixed w={fixed_w:.4f}",
        fontsize=12,
    )
    fig.savefig(corner_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
# =============================================================================
# Parameter bounds (4 params: gamma, omega, t_E_aff, del_go)
# =============================================================================
omega_bounds = [0.1, 15]
omega_plausible_bounds = [2, 12]

t_E_aff_bounds = [0, 1]
t_E_aff_plausible_bounds = [0.01, 0.2]

del_go_bounds = [0.001, 0.2]
del_go_plausible_bounds = [0.11, 0.15]


# %%
# =============================================================================
# Main fitting loop
# =============================================================================
total_completed = 0
total_existing = 0
total_corners_generated = 0
total_low_trial_skips = 0
total_errors = 0

for batch_name, animal_id in batch_animal_pairs:
    animal_key = (batch_name, str(animal_id))
    fixed_w = w_by_batch_animal.get(animal_key)

    print("\n" + "=" * 70)
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print("=" * 70)

    if fixed_w is None:
        print(f"  WARNING: No mean_w found in {W_SUMMARY_CSV_PATH}. Skipping animal.")
        total_errors += 1
        continue

    abort_params = get_abort_params_from_animal_pkl_file(batch_name, animal_id)
    if abort_params is None:
        print(f"  WARNING: No abort params found for {batch_name}/{animal_id}. Skipping animal.")
        total_errors += 1
        continue

    V_A = abort_params["V_A"]
    theta_A = abort_params["theta_A"]
    t_A_aff = abort_params["t_A_aff"]

    batch_file = os.path.join(batch_dir, f"batch_{batch_name}_valid_and_aborts.csv")
    df = pd.read_csv(batch_file)
    df_animal = df[df["animal"] == int(animal_id)]
    df_animal_success = df_animal[df_animal["success"].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[
        (df_animal_success["RTwrtStim"] <= 1) & (df_animal_success["RTwrtStim"] > 0)
    ]

    print(f"  fixed mean_w: {fixed_w:.6f}")
    print(f"  valid RT-filtered trials: {len(df_animal_success_rt_filter)}")

    animal_completed = 0
    animal_existing = 0
    animal_corners_generated = 0
    animal_low_trial_skips = 0
    animal_errors = 0

    for cond_ABL, cond_ILD in conditions_to_fit:
        pkl_file = os.path.join(
            OUTPUT_FOLDER,
            f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_FIX_w_mean_4_params.pkl",
        )
        corner_plot_file = os.path.join(
            CORNER_PLOT_FOLDER,
            f"corner_cond_by_cond_{batch_name}_{animal_id}_{cond_ABL}_ILD_{cond_ILD}_FIX_w_mean_4_params.png",
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
                    animal_errors += 1
                    total_errors += 1
            continue

        print(f"  [{cond_ABL}, {cond_ILD:+3d}] Fitting with {n_trials} trials...")

        def compute_loglike_trial(row, gamma, omega, t_E_aff, del_go):
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
                del_go,
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
            gamma, omega, t_E_aff, del_go = params
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
            del_go_logpdf = trapezoidal_logpdf(
                del_go,
                del_go_bounds[0],
                del_go_plausible_bounds[0],
                del_go_plausible_bounds[1],
                del_go_bounds[1],
            )
            return gamma_logpdf + omega_logpdf + t_E_aff_logpdf + del_go_logpdf

        def vbmc_loglike_fn(params):
            gamma, omega, t_E_aff, del_go = params
            all_loglike = Parallel(n_jobs=N_JOBS)(
                delayed(compute_loglike_trial)(row, gamma, omega, t_E_aff, del_go)
                for _, row in df_cond.iterrows()
            )
            return np.sum(all_loglike)

        def vbmc_joint_fn(params):
            return vbmc_prior_fn(params) + vbmc_loglike_fn(params)

        lb = np.array([gamma_bounds[0], omega_bounds[0], t_E_aff_bounds[0], del_go_bounds[0]])
        ub = np.array([gamma_bounds[1], omega_bounds[1], t_E_aff_bounds[1], del_go_bounds[1]])
        plb = np.array(
            [
                gamma_plausible_bounds[0],
                omega_plausible_bounds[0],
                t_E_aff_plausible_bounds[0],
                del_go_plausible_bounds[0],
            ]
        )
        pub = np.array(
            [
                gamma_plausible_bounds[1],
                omega_plausible_bounds[1],
                t_E_aff_plausible_bounds[1],
                del_go_plausible_bounds[1],
            ]
        )

        np.random.seed(42)
        x_0 = np.array(
            [
                np.random.uniform(gamma_plausible_bounds[0], gamma_plausible_bounds[1]),
                np.random.uniform(omega_plausible_bounds[0], omega_plausible_bounds[1]),
                np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1]),
                np.random.uniform(del_go_plausible_bounds[0], del_go_plausible_bounds[1]),
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
                batch_name,
                animal_id,
                cond_ABL,
                cond_ILD,
            )
            print(f"    -> Saved corner: {os.path.basename(corner_plot_file)}")
            animal_corners_generated += 1
            total_corners_generated += 1
        except Exception as e:
            print(f"    -> ERROR: {e}")
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
    f"low_trial_skips={total_low_trial_skips}, errors={total_errors}"
)
print("=" * 70)

# %%
