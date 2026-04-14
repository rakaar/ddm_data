# %%
"""
Fit normalized TIED parameters on LED-OFF data using proactive+lapse parameters loaded from
proactive VP pickles.

This variant uses true right-truncation at {cut off=1 s} after stimulus onset, models the
evidence-afferent delay t_E_aff as a linear function of ABL and abs_ILD in milliseconds,
rejects parameter sets whose implied delay is negative for any observed fit condition, and
uses the observed trial choice in the likelihood:
    delay_ms = bias_ms + abl_coeff_ms_per_abl * ABL + abs_ild_coeff_ms_per_unit * abs_ILD
    require delay_ms >= 0 on observed (ABL, abs_ILD) conditions used in the fit
    t_E_aff = delay_ms * 1e-3
    RT density = pdf(bound=observed choice)
"""

# %%
from pathlib import Path
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyvbmc import VBMC

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANIMAL_WISE_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(ANIMAL_WISE_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_WISE_DIR))

from proactive_plus_lapse_plus_reactive_uitls import (
    trial_logpdf_proactive_lapse_only_no_trunc_right_truncated,
)
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from animal_wise_config import T_trunc


# %%
############ Parameters (edit here) ############
batch_name = "LED7"
fit_mode = "aggregate"  # "aggregate" or "animal_wise"
ANIMAL_ID = None  # Used only when fit_mode == "animal_wise"
animals_to_fit = None  # Used only when fit_mode == "animal_wise"
save_corner_plot = True

session_type = 7
training_level = 16
allowed_repeat_trials = [0, 2]

max_rtwrtstim_for_fit = 1.0
n_jobs = 30
posterior_sample_count = int(1e5)
proactive_posterior_sample_count = int(2e4)
vbmc_max_fun_evals = 200 * (2 + 9)

led_data_csv_path = REPO_ROOT / "out_LED.csv"
proactive_vp_template = SCRIPT_DIR / "vbmc_real_animal_{animal}_fit_NO_TRUNC_with_lapse.pkl"
proactive_vp_aggregate_path = SCRIPT_DIR / "vbmc_real_all_animals_fit_NO_TRUNC_with_lapse.pkl"

output_dir = SCRIPT_DIR / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_linear_delay_with_choice"
output_dir.mkdir(parents=True, exist_ok=True)

# ###### RUN TAG ######
K_max = 10
is_norm = True
is_time_vary = False
phi_params_obj = np.nan
truncate_rt_wrt_stim_s = 1.0
supported_abl_values = (20, 40, 60)

if truncate_rt_wrt_stim_s <= 0:
    raise ValueError("truncate_rt_wrt_stim_s must be positive.")
if truncate_rt_wrt_stim_s > max_rtwrtstim_for_fit:
    raise ValueError("truncate_rt_wrt_stim_s cannot exceed max_rtwrtstim_for_fit.")

truncate_ms = int(round(float(truncate_rt_wrt_stim_s) * 1e3))
run_tag = f"trunc{truncate_ms}ms"


# %%
############ Global vars used by likelihood callbacks ############
V_A = np.nan
theta_A = np.nan
t_A_aff = np.nan
lapse_prob = np.nan
beta_lapse = np.nan
df_valid_animal_truncated = pd.DataFrame()
observed_delay_condition_pairs = np.empty((0, 2), dtype=float)


# %%
############ Helpers ############
def get_t_E_aff_from_abl_abs_ild(
    abl,
    abs_ild,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
):
    delay_ms = (
        float(bias_ms)
        + float(abl_delay_coeff_ms_per_abl) * float(abl)
        + float(abs_ild_delay_coeff_ms_per_unit) * float(abs_ild)
    )
    return delay_ms * 1e-3


def validate_supported_abl_values(df, df_name):
    observed = np.sort(df["ABL"].dropna().astype(float).unique())
    if len(observed) == 0:
        raise ValueError(f"No ABL values found in {df_name}.")

    unexpected = [
        float(abl)
        for abl in observed
        if not any(np.isclose(float(abl), float(supported)) for supported in supported_abl_values)
    ]
    if unexpected:
        raise ValueError(
            f"Unexpected ABL values in {df_name}: {unexpected}. "
            f"Supported values are {supported_abl_values}."
        )
    return observed


def format_abl_counts(df):
    counts = (
        df["ABL"]
        .astype(float)
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {int(k): int(v) for k, v in counts.items()}


# %%
########### Normalized TIED likelihood/prior ##############
def compute_loglike_norm_fn(
    row,
    rate_lambda,
    T_0,
    theta_E,
    Z_E,
    bias_ms,
    abl_delay_coeff_ms_per_abl,
    abs_ild_delay_coeff_ms_per_unit,
    del_go,
    rate_norm_l,
):
    t_E_aff = get_t_E_aff_from_abl_abs_ild(
        abl=row["ABL"],
        abs_ild=row["abs_ILD"],
        bias_ms=bias_ms,
        abl_delay_coeff_ms_per_abl=abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit=abs_ild_delay_coeff_ms_per_unit,
    )
    return trial_logpdf_proactive_lapse_only_no_trunc_right_truncated(
        row=row,
        V_A=V_A,
        theta_A=theta_A,
        t_A_aff=t_A_aff,
        rate_lambda=rate_lambda,
        T0=T_0,
        theta_E=theta_E,
        Z_E=Z_E,
        t_E_aff=t_E_aff,
        del_go=del_go,
        phi_params=phi_params_obj,
        rate_norm_l=rate_norm_l,
        is_norm=is_norm,
        is_time_vary=is_time_vary,
        K_max=K_max,
        lapse_prob=lapse_prob,
        beta_lapse=beta_lapse,
        lapse_choice_prob=0.5,
        truncate_rt_wrt_stim=truncate_rt_wrt_stim_s,
        eps=1e-50,
    )


def vbmc_norm_tied_loglike_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        del_go,
        rate_norm_l,
    ) = params
    delay_ms_by_condition = (
        float(bias_ms)
        + float(abl_delay_coeff_ms_per_abl) * observed_delay_condition_pairs[:, 0]
        + float(abs_ild_delay_coeff_ms_per_unit) * observed_delay_condition_pairs[:, 1]
    )
    if np.any(delay_ms_by_condition < 0):
        return np.log(1e-50)

    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=n_jobs)(
        delayed(compute_loglike_norm_fn)(
            row,
            rate_lambda,
            T_0,
            theta_E,
            Z_E,
            bias_ms,
            abl_delay_coeff_ms_per_abl,
            abs_ild_delay_coeff_ms_per_unit,
            del_go,
            rate_norm_l,
        )
        for _, row in df_valid_animal_truncated.iterrows()
    )
    return np.sum(all_loglike)


def vbmc_prior_norm_tied_fn(params):
    (
        rate_lambda,
        T_0,
        theta_E,
        w,
        bias_ms,
        abl_delay_coeff_ms_per_abl,
        abs_ild_delay_coeff_ms_per_unit,
        del_go,
        rate_norm_l,
    ) = params
    return (
        trapezoidal_logpdf(
            rate_lambda,
            norm_tied_rate_lambda_bounds[0],
            norm_tied_rate_lambda_plausible_bounds[0],
            norm_tied_rate_lambda_plausible_bounds[1],
            norm_tied_rate_lambda_bounds[1],
        )
        + trapezoidal_logpdf(
            T_0,
            norm_tied_T_0_bounds[0],
            norm_tied_T_0_plausible_bounds[0],
            norm_tied_T_0_plausible_bounds[1],
            norm_tied_T_0_bounds[1],
        )
        + trapezoidal_logpdf(
            theta_E,
            norm_tied_theta_E_bounds[0],
            norm_tied_theta_E_plausible_bounds[0],
            norm_tied_theta_E_plausible_bounds[1],
            norm_tied_theta_E_bounds[1],
        )
        + trapezoidal_logpdf(
            w,
            norm_tied_w_bounds[0],
            norm_tied_w_plausible_bounds[0],
            norm_tied_w_plausible_bounds[1],
            norm_tied_w_bounds[1],
        )
        + trapezoidal_logpdf(
            bias_ms,
            norm_tied_bias_ms_bounds[0],
            norm_tied_bias_ms_plausible_bounds[0],
            norm_tied_bias_ms_plausible_bounds[1],
            norm_tied_bias_ms_bounds[1],
        )
        + trapezoidal_logpdf(
            abl_delay_coeff_ms_per_abl,
            norm_tied_abl_delay_coeff_ms_per_abl_bounds[0],
            norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[0],
            norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[1],
            norm_tied_abl_delay_coeff_ms_per_abl_bounds[1],
        )
        + trapezoidal_logpdf(
            abs_ild_delay_coeff_ms_per_unit,
            norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[0],
            norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[0],
            norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[1],
            norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[1],
        )
        + trapezoidal_logpdf(
            del_go,
            norm_tied_del_go_bounds[0],
            norm_tied_del_go_plausible_bounds[0],
            norm_tied_del_go_plausible_bounds[1],
            norm_tied_del_go_bounds[1],
        )
        + trapezoidal_logpdf(
            rate_norm_l,
            norm_tied_rate_norm_bounds[0],
            norm_tied_rate_norm_plausible_bounds[0],
            norm_tied_rate_norm_plausible_bounds[1],
            norm_tied_rate_norm_bounds[1],
        )
    )


def vbmc_norm_tied_joint_fn(params):
    return vbmc_prior_norm_tied_fn(params) + vbmc_norm_tied_loglike_fn(params)


############ Bounds ############
norm_tied_rate_lambda_bounds = [0.5, 5]
norm_tied_T_0_bounds = [10e-3, 500e-3]
norm_tied_theta_E_bounds = [1, 15]
norm_tied_w_bounds = [0.3, 0.7]
norm_tied_bias_ms_bounds = [10, 200]
norm_tied_abl_delay_coeff_ms_per_abl_bounds = [-1, 0.5]
norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds = [-2, 0.5]
norm_tied_del_go_bounds = [0, 0.2]
norm_tied_rate_norm_bounds = [0, 2]

norm_tied_rate_lambda_plausible_bounds = [1, 3]
norm_tied_T_0_plausible_bounds = [20e-3, 50e-3]
norm_tied_theta_E_plausible_bounds = [1.5, 10]
norm_tied_w_plausible_bounds = [0.4, 0.6]
norm_tied_bias_ms_plausible_bounds = [50, 150]
norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds = [-0.75, -0.1]
norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds = [-1, -0.1]
norm_tied_del_go_plausible_bounds = [0.05, 0.15]
norm_tied_rate_norm_plausible_bounds = [0.8, 0.99]

norm_tied_lb = np.array([
    norm_tied_rate_lambda_bounds[0],
    norm_tied_T_0_bounds[0],
    norm_tied_theta_E_bounds[0],
    norm_tied_w_bounds[0],
    norm_tied_bias_ms_bounds[0],
    norm_tied_abl_delay_coeff_ms_per_abl_bounds[0],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[0],
    norm_tied_del_go_bounds[0],
    norm_tied_rate_norm_bounds[0],
])

norm_tied_ub = np.array([
    norm_tied_rate_lambda_bounds[1],
    norm_tied_T_0_bounds[1],
    norm_tied_theta_E_bounds[1],
    norm_tied_w_bounds[1],
    norm_tied_bias_ms_bounds[1],
    norm_tied_abl_delay_coeff_ms_per_abl_bounds[1],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_bounds[1],
    norm_tied_del_go_bounds[1],
    norm_tied_rate_norm_bounds[1],
])

norm_tied_plb = np.array([
    norm_tied_rate_lambda_plausible_bounds[0],
    norm_tied_T_0_plausible_bounds[0],
    norm_tied_theta_E_plausible_bounds[0],
    norm_tied_w_plausible_bounds[0],
    norm_tied_bias_ms_plausible_bounds[0],
    norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[0],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[0],
    norm_tied_del_go_plausible_bounds[0],
    norm_tied_rate_norm_plausible_bounds[0],
])

norm_tied_pub = np.array([
    norm_tied_rate_lambda_plausible_bounds[1],
    norm_tied_T_0_plausible_bounds[1],
    norm_tied_theta_E_plausible_bounds[1],
    norm_tied_w_plausible_bounds[1],
    norm_tied_bias_ms_plausible_bounds[1],
    norm_tied_abl_delay_coeff_ms_per_abl_plausible_bounds[1],
    norm_tied_abs_ild_delay_coeff_ms_per_unit_plausible_bounds[1],
    norm_tied_del_go_plausible_bounds[1],
    norm_tied_rate_norm_plausible_bounds[1],
])


# %%
############ Load + preprocess LED7-style data ############
if not led_data_csv_path.exists():
    raise FileNotFoundError(f"Could not find data CSV: {led_data_csv_path}")

exp_df = pd.read_csv(led_data_csv_path)
exp_df["RTwrtStim"] = exp_df["timed_fix"] - exp_df["intended_fix"]
exp_df = exp_df.rename(columns={"timed_fix": "TotalFixTime"})
exp_df["abs_ILD"] = exp_df["ILD"].abs()
exp_df = exp_df[exp_df["RTwrtStim"] <= max_rtwrtstim_for_fit]
exp_df = exp_df[~((exp_df["RTwrtStim"].isna()) & (exp_df["abort_event"] == 3))].copy()

mask_nan = exp_df["response_poke"].isna()
mask_success_1 = exp_df["success"] == 1
mask_success_neg1 = exp_df["success"] == -1
mask_ild_pos = exp_df["ILD"] > 0
mask_ild_neg = exp_df["ILD"] < 0
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, "response_poke"] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, "response_poke"] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, "response_poke"] = 3

mask_led_off = (exp_df["LED_trial"] == 0) | (exp_df["LED_trial"].isna())
mask_repeat = exp_df["repeat_trial"].isin(allowed_repeat_trials) | exp_df["repeat_trial"].isna()
exp_df_led_off = exp_df[
    mask_led_off
    & mask_repeat
    & exp_df["session_type"].isin([session_type])
    & exp_df["training_level"].isin([training_level])
].copy()

exp_df_led_off["choice"] = np.where(
    exp_df_led_off["response_poke"] == 3,
    1,
    np.where(exp_df_led_off["response_poke"] == 2, -1, np.nan),
)
missing_choice = exp_df_led_off["choice"].isna()
if missing_choice.any():
    exp_df_led_off.loc[missing_choice, "choice"] = np.random.choice(
        [1, -1], size=int(missing_choice.sum())
    )
exp_df_led_off["choice"] = exp_df_led_off["choice"].astype(int)

df_valid_and_aborts = exp_df_led_off[
    (exp_df_led_off["success"].isin([1, -1])) | (exp_df_led_off["abort_event"] == 3)
].copy()
all_animals = np.sort(df_valid_and_aborts["animal"].unique())
observed_abl_values = validate_supported_abl_values(
    df_valid_and_aborts, "filtered LED-OFF valid+aborts dataset"
)
observed_abs_ild_values = np.sort(df_valid_and_aborts["abs_ILD"].dropna().astype(float).unique())

print(
    f"Batch={batch_name}, session_type={session_type}, training_level={training_level}, "
    f"LED OFF trials only"
)
print(
    f"T_trunc={T_trunc} (not used in likelihood), animals in filtered data={all_animals}"
)
print(
    "Right truncation: keep trials with "
    f"0 < RTwrtStim <= {truncate_rt_wrt_stim_s:.3f}s, ignore later valid trials during fitting, "
    "and use the observed trial choice in the bound-specific likelihood."
)
print(f"Run tag for saved fit artifacts: {run_tag}")
print(f"Supported ABL values in filtered data: {observed_abl_values.tolist()}")
print(f"Observed abs_ILD values in filtered data: {observed_abs_ild_values.tolist()}")
print(f"Filtered LED-OFF trial counts by ABL: {format_abl_counts(df_valid_and_aborts)}")


# %%
def fit_one_unit(unit_tag, df_unit, proactive_vp_path):
    global V_A, theta_A, t_A_aff, lapse_prob, beta_lapse, df_valid_animal_truncated
    global observed_delay_condition_pairs

    if not proactive_vp_path.exists():
        print(f"Skipping {unit_tag}: proactive VP pickle not found: {proactive_vp_path}")
        return

    observed_unit_abl_values = validate_supported_abl_values(df_unit, f"{unit_tag} filtered dataset")
    observed_unit_abs_ild_values = np.sort(df_unit["abs_ILD"].dropna().astype(float).unique())
    print(f"{unit_tag}: observed ABL values -> {observed_unit_abl_values.tolist()}")
    print(f"{unit_tag}: observed abs_ILD values -> {observed_unit_abs_ild_values.tolist()}")

    with open(proactive_vp_path, "rb") as f:
        proactive_vp = pickle.load(f)

    proactive_samples = proactive_vp.sample(proactive_posterior_sample_count)[0]
    V_A_base = float(np.mean(proactive_samples[:, 0]))
    theta_A = float(np.mean(proactive_samples[:, 2]))
    del_a_minus_del_LED = float(np.mean(proactive_samples[:, 3]))
    del_m_plus_del_LED = float(np.mean(proactive_samples[:, 4]))
    lapse_prob = float(np.mean(proactive_samples[:, 5]))
    beta_lapse = float(np.mean(proactive_samples[:, 6]))

    V_A = V_A_base
    t_A_aff = del_a_minus_del_LED + del_m_plus_del_LED
    print(
        f"\n{unit_tag}: loaded proactive params -> "
        f"V_A_base={V_A_base:.4f}, theta_A={theta_A:.4f}, "
        f"del_a_minus_del_LED={del_a_minus_del_LED:.4f}, del_m_plus_del_LED={del_m_plus_del_LED:.4f}, "
        f"lapse_prob={lapse_prob:.4f}, beta_lapse={beta_lapse:.4f}, "
        f"derived t_A_aff={t_A_aff:.4f}"
    )

    df_valid_unit = df_unit[df_unit["success"].isin([1, -1])].copy()
    df_valid_unit_for_fit_window = df_valid_unit[
        (df_valid_unit["RTwrtStim"] > 0) & (df_valid_unit["RTwrtStim"] <= max_rtwrtstim_for_fit)
    ].copy()
    df_valid_animal_truncated = df_valid_unit_for_fit_window[
        df_valid_unit_for_fit_window["RTwrtStim"] <= truncate_rt_wrt_stim_s
    ].copy()

    invalid_choice_mask = ~df_valid_animal_truncated["choice"].isin([-1, 1])
    if invalid_choice_mask.any():
        invalid_choices = (
            df_valid_animal_truncated.loc[invalid_choice_mask, "choice"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"{unit_tag}: fit rows must have valid choices in {{-1, +1}}. "
            f"Found invalid choice values: {invalid_choices}"
        )

    pre_trunc_valid_trial_count = len(df_valid_unit_for_fit_window)
    post_trunc_valid_trial_count = len(df_valid_animal_truncated)
    ignored_valid_trial_count = pre_trunc_valid_trial_count - post_trunc_valid_trial_count
    abl_counts_before = format_abl_counts(df_valid_unit_for_fit_window)
    abl_counts_after = format_abl_counts(df_valid_animal_truncated)

    if post_trunc_valid_trial_count == 0:
        print(f"Skipping {unit_tag}: no valid trials remain after right truncation")
        return

    print(
        f"{unit_tag}: valid trials before truncation={pre_trunc_valid_trial_count}, "
        f"after truncation={post_trunc_valid_trial_count}, ignored={ignored_valid_trial_count}"
    )
    print(
        f"{unit_tag}: ABL counts before truncation={abl_counts_before}, "
        f"after truncation={abl_counts_after}"
    )
    print(
        f"{unit_tag}: fitting uses the observed trial choice in the likelihood."
    )

    observed_delay_condition_pairs = (
        df_valid_animal_truncated[["ABL", "abs_ILD"]]
        .drop_duplicates()
        .sort_values(["ABL", "abs_ILD"])
        .to_numpy(dtype=float)
    )
    print(
        f"{unit_tag}: observed delay conditions used for fit -> "
        f"{observed_delay_condition_pairs.tolist()}"
    )

    x_0 = np.array([2.3, 100e-3, 3.0, 0.51, 100.0, -0.425, -0.55, 0.13, 0.95])
    x_0 = np.clip(x_0, norm_tied_plb, norm_tied_pub)

    vbmc = VBMC(
        vbmc_norm_tied_joint_fn,
        x_0,
        norm_tied_lb,
        norm_tied_ub,
        norm_tied_plb,
        norm_tied_pub,
        options={"display": "on", "max_fun_evals": vbmc_max_fun_evals},
    )
    vp, results = vbmc.optimize()

    if unit_tag == "aggregate":
        base_name = (
            f"batch_{batch_name}_aggregate_ledoff_1_proactive_loaded_truncate_NOT_censor_linear_delay_with_choice_{run_tag}"
        )
    else:
        base_name = (
            f"batch_{batch_name}_animal_{unit_tag}_ledoff_1_"
            f"proactive_loaded_truncate_NOT_censor_linear_delay_with_choice_{run_tag}"
        )

    vbmc_obj_path = output_dir / f"vbmc_norm_tied_{base_name}.pkl"
    vp.save(str(vbmc_obj_path), overwrite=True)

    vp_samples = vp.sample(posterior_sample_count)[0]
    rate_lambda = float(vp_samples[:, 0].mean())
    T_0 = float(vp_samples[:, 1].mean())
    theta_E = float(vp_samples[:, 2].mean())
    w = float(vp_samples[:, 3].mean())
    bias_ms = float(vp_samples[:, 4].mean())
    abl_delay_coeff_ms_per_abl = float(vp_samples[:, 5].mean())
    abs_ild_delay_coeff_ms_per_unit = float(vp_samples[:, 6].mean())
    del_go = float(vp_samples[:, 7].mean())
    rate_norm_l = float(vp_samples[:, 8].mean())
    Z_E = (w - 0.5) * 2 * theta_E

    norm_tied_loglike = vbmc_norm_tied_loglike_fn(
        [
            rate_lambda,
            T_0,
            theta_E,
            w,
            bias_ms,
            abl_delay_coeff_ms_per_abl,
            abs_ild_delay_coeff_ms_per_unit,
            del_go,
            rate_norm_l,
        ]
    )

    posterior_mean_delay_ms_by_abl_abs_ild = {
        int(abl): {
            int(abs_ild): 1e3
            * get_t_E_aff_from_abl_abs_ild(
                abl=abl,
                abs_ild=abs_ild,
                bias_ms=bias_ms,
                abl_delay_coeff_ms_per_abl=abl_delay_coeff_ms_per_abl,
                abs_ild_delay_coeff_ms_per_unit=abs_ild_delay_coeff_ms_per_unit,
            )
            for abs_ild in observed_unit_abs_ild_values
        }
        for abl in supported_abl_values
    }

    corner_path = None
    if save_corner_plot:
        corner_samples = vp_samples.copy()
        corner_samples[:, 1] *= 1e3
        labels = [
            r"$\lambda$",
            r"$T_0$ (ms)",
            r"$\theta_E$",
            r"$w$",
            "bias (ms)",
            "ABL coeff (ms/ABL)",
            "|ILD| coeff (ms/unit)",
            r"$\Delta_{go}$",
            "rate_norm_l",
        ]
        corner_fig = corner.corner(
            corner_samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.025, 0.5, 0.975],
            title_fmt=".3f",
        )
        corner_fig.suptitle(f"Normalized TIED Posterior ({unit_tag}, linear delay, with choice)", y=1.02)
        corner_path = output_dir / f"corner_norm_tied_{base_name}.pdf"
        corner_fig.savefig(corner_path, bbox_inches="tight")
        print(f"Saved corner plot: {corner_path}")

    print("Posterior means:")
    print(
        f"rate_lambda={rate_lambda:.5f}, T_0(ms)={1e3*T_0:.5f}, theta_E={theta_E:.5f}, "
        f"w={w:.5f}, Z_E={Z_E:.5f}, bias_ms={bias_ms:.5f}, "
        f"abl_delay_coeff_ms_per_abl={abl_delay_coeff_ms_per_abl:.5f}, "
        f"abs_ild_delay_coeff_ms_per_unit={abs_ild_delay_coeff_ms_per_unit:.5f}, "
        f"del_go={del_go:.5f}, rate_norm_l={rate_norm_l:.5f}"
    )

    vbmc_norm_tied_results = {
        "rate_lambda_samples": vp_samples[:, 0],
        "T_0_samples": vp_samples[:, 1],
        "theta_E_samples": vp_samples[:, 2],
        "w_samples": vp_samples[:, 3],
        "bias_ms_samples": vp_samples[:, 4],
        "abl_delay_coeff_ms_per_abl_samples": vp_samples[:, 5],
        "abs_ild_delay_coeff_ms_per_unit_samples": vp_samples[:, 6],
        "del_go_samples": vp_samples[:, 7],
        "rate_norm_l_samples": vp_samples[:, 8],
        "message": results.get("message"),
        "elbo": results.get("elbo"),
        "elbo_sd": results.get("elbo_sd"),
        "loglike": norm_tied_loglike,
    }

    fit_trial_counts = {
        "valid_trials_before_right_truncation": pre_trunc_valid_trial_count,
        "valid_trials_after_right_truncation": post_trunc_valid_trial_count,
        "ignored_valid_trials_above_right_truncation": ignored_valid_trial_count,
        "abl_counts_before_right_truncation": abl_counts_before,
        "abl_counts_after_right_truncation": abl_counts_after,
        "valid_trials_used_for_fit": post_trunc_valid_trial_count,
        "abl_counts_used_for_fit": dict(abl_counts_after),
    }

    out_results_path = output_dir / f"results_norm_tied_{base_name}.pkl"
    save_dict = {
        "unit_tag": unit_tag,
        "source_proactive_vp_pkl": str(proactive_vp_path),
        "fit_config": {
            "batch_name": batch_name,
            "fit_mode": fit_mode,
            "session_type": session_type,
            "training_level": training_level,
            "allowed_repeat_trials": allowed_repeat_trials,
            "led_data_csv_path": str(led_data_csv_path),
            "likelihood_mode": "proactive_lapse_only_no_trunc_right_truncated_linear_delay_with_choice",
            "right_truncation_rule": (
                "fit only trials with 0 < RT - t_stim <= truncate_rt_wrt_stim_s; "
                "use the observed-choice bound-specific RT density; "
                "normalize retained trial likelihood by "
                "CDF(t_stim + truncate_rt_wrt_stim_s) - CDF(t_stim)"
            ),
            "choice_mode": "use_observed_choice",
            "delay_rule_ms": (
                "bias_ms + abl_coeff_ms_per_abl * ABL + "
                "abs_ild_coeff_ms_per_unit * abs_ILD"
            ),
            "negative_delay_rule": "if delay_ms < 0 for any observed fit condition, return log(1e-50)",
            "delay_return_units": "seconds",
            "parameter_order": [
                "rate_lambda",
                "T_0",
                "theta_E",
                "w",
                "bias_ms",
                "abl_delay_coeff_ms_per_abl",
                "abs_ild_delay_coeff_ms_per_unit",
                "del_go",
                "rate_norm_l",
            ],
            "supported_ABL_values": list(supported_abl_values),
            "observed_abs_ILD_values": observed_unit_abs_ild_values.tolist(),
            "observed_delay_condition_pairs_used_for_fit": observed_delay_condition_pairs.tolist(),
            "truncate_rt_wrt_stim_s": truncate_rt_wrt_stim_s,
            "run_tag": run_tag,
            "proactive_vp_template": str(proactive_vp_template),
            "proactive_vp_aggregate_path": str(proactive_vp_aggregate_path),
            "proactive_posterior_sample_count": proactive_posterior_sample_count,
            "max_rtwrtstim_for_fit": max_rtwrtstim_for_fit,
            "save_corner_plot": save_corner_plot,
            "corner_path": str(corner_path) if corner_path is not None else None,
            "T_trunc": T_trunc,
        },
        "fit_trial_counts": fit_trial_counts,
        "loaded_proactive_params": {
            "V_A_base": V_A_base,
            "theta_A": theta_A,
            "del_a_minus_del_LED": del_a_minus_del_LED,
            "del_m_plus_del_LED": del_m_plus_del_LED,
            "lapse_prob": lapse_prob,
            "beta_lapse": beta_lapse,
            "derived_t_A_aff_for_tied_fit": t_A_aff,
        },
        "vbmc_norm_tied_results": vbmc_norm_tied_results,
        "posterior_mean_delay_ms_by_abl_abs_ild": posterior_mean_delay_ms_by_abl_abs_ild,
    }
    with open(out_results_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved VBMC object: {vbmc_obj_path}")
    print(f"Saved results: {out_results_path}")


# %%
############ Run mode ############
if fit_mode == "aggregate":
    print("Running aggregate fit (single fit across all filtered LED-OFF trials) with linear-delay choice-aware likelihood.")
    fit_one_unit(
        unit_tag="aggregate",
        df_unit=df_valid_and_aborts,
        proactive_vp_path=proactive_vp_aggregate_path,
    )
elif fit_mode == "animal_wise":
    if ANIMAL_ID is not None:
        if ANIMAL_ID < 0 or ANIMAL_ID >= len(all_animals):
            raise IndexError(
                f"ANIMAL_ID={ANIMAL_ID} out of range for {len(all_animals)} animals: {all_animals}"
            )
        selected = [int(all_animals[ANIMAL_ID])]
    elif animals_to_fit is None:
        selected = [int(a) for a in all_animals.tolist()]
    else:
        selected = [int(a) for a in animals_to_fit]

    print(f"Running animal-wise linear-delay choice-aware fit for animals: {selected}")
    for animal in selected:
        if animal not in all_animals:
            print(f"Skipping animal {animal}: not found in filtered data")
            continue
        df_one = df_valid_and_aborts[df_valid_and_aborts["animal"] == animal].copy()
        fit_one_unit(
            unit_tag=str(animal),
            df_unit=df_one,
            proactive_vp_path=Path(str(proactive_vp_template).format(animal=animal)),
        )
else:
    raise ValueError(f"Unknown fit_mode: {fit_mode}. Use 'aggregate' or 'animal_wise'.")

print("\nDone.")
