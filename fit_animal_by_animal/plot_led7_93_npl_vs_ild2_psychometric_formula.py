# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

import figure_template as ft
from time_vary_norm_alpha_utils import rho_E_minus_small_t_NORM_alpha_time_varying_fn


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

BATCH_NAME = "LED7"
ANIMAL_ID = 93
ANIMAL_KEY = (BATCH_NAME, ANIMAL_ID)
ABL_ARR = [20, 40, 60]
ILD_CONTINUOUS = np.round(np.arange(-16, 16.0001, 0.1), 1)
T_PTS_REACTIVE_ONLY = np.arange(0, 1.0001, 0.001)
REACTIVE_ONLY_W = 0.6
K_MAX = 10

NPL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_PSY_PKL = os.path.join(
    REPO_DIR,
    "NPL_alpha_ILD2_fit_results",
    "figure_4_diagnostics",
    "theoretical_psychometric_data_npl_alpha_ild2.pkl",
)
NPL_RESULT_PKL = os.path.join(SCRIPT_DIR, f"results_{BATCH_NAME}_animal_{ANIMAL_ID}.pkl")
ILD2_RESULT_PKL = os.path.join(
    REPO_DIR,
    "NPL_alpha_ILD2_fit_results",
    "result_pkls",
    f"results_{BATCH_NAME}_animal_{ANIMAL_ID}_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS.pkl",
)

OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, f"{BATCH_NAME}_{ANIMAL_ID}_npl_vs_ild2_psychometric_formula.png")


# %%
def sigmoid_from_fit_params(ild_values, params):
    upper, lower, x0, slope = params
    return lower + (upper - lower) / (1 + np.exp(-slope * (ild_values - x0)))


def gamma_alpha_formula_psychometric(ild_values, theta, rate_lambda, rate_norm_l, alpha):
    chi = 17.37
    ild_arg = rate_lambda * ild_values / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ild_values / chi
    log_rate_ratio = (
        2 * ild_arg
        + np.log1p(alpha * np.exp(2 * norm_ild_arg))
        - np.log(np.exp(2 * norm_ild_arg) + alpha)
    )
    gamma = theta * np.tanh(log_rate_ratio / 2)
    return 1 / (1 + np.exp(-2 * gamma))


def delay_s_from_params(ABL, ILD, params):
    if "t_E_aff" in params:
        return params["t_E_aff"]

    delay_ms = (
        params["bias_ms"]
        + params["abl_delay_coeff_ms_per_abl"] * ABL
        + params["abs_ild_delay_coeff_ms_per_unit"] * abs(ILD)
        + params["ild2_delay_coeff_ms_per_unit2"] * (ILD ** 2)
    )
    return delay_ms * 1e-3


def get_posterior_mean_params(result_path, result_key):
    with open(result_path, "rb") as handle:
        fit_results = pickle.load(handle)
    model_results = fit_results[result_key]
    params = {
        "theta_E": float(np.mean(model_results["theta_E_samples"])),
        "rate_lambda": float(np.mean(model_results["rate_lambda_samples"])),
        "T_0": float(np.mean(model_results["T_0_samples"])),
        "rate_norm_l": float(np.mean(model_results.get("rate_norm_l_samples", [0.0]))),
        "alpha": float(np.mean(model_results.get("alpha_samples", [1.0]))),
    }
    if "t_E_aff_samples" in model_results:
        params["t_E_aff"] = float(np.mean(model_results["t_E_aff_samples"]))
    for sample_key, param_key in [
        ("bias_ms_samples", "bias_ms"),
        ("abl_delay_coeff_ms_per_abl_samples", "abl_delay_coeff_ms_per_abl"),
        ("abs_ild_delay_coeff_ms_per_unit_samples", "abs_ild_delay_coeff_ms_per_unit"),
        ("ild2_delay_coeff_ms_per_unit2_samples", "ild2_delay_coeff_ms_per_unit2"),
    ]:
        if sample_key in model_results:
            params[param_key] = float(np.mean(model_results[sample_key]))
    return params


def average_sigmoid_slope(psychometric_data):
    slopes = []
    model_data = psychometric_data[ANIMAL_KEY]
    for ABL in ABL_ARR:
        fit = model_data[ABL].get("fit")
        if fit is not None and "params" in fit:
            slopes.append(float(fit["params"][3]))
    return float(np.nanmean(slopes))


def reactive_only_psychometric(ild_values, ABL, params):
    Z_E = (REACTIVE_ONLY_W - 0.5) * 2 * params["theta_E"]
    right_choice_probs = []

    for ILD in ild_values:
        t_E_aff = delay_s_from_params(ABL, ILD, params)
        up_density = np.array([
            rho_E_minus_small_t_NORM_alpha_time_varying_fn(
                t - t_E_aff,
                1,
                ABL,
                ILD,
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                np.nan,
                np.nan,
                params["rate_norm_l"],
                params["alpha"],
                True,
                False,
                K_MAX,
            )
            for t in T_PTS_REACTIVE_ONLY
        ])
        down_density = np.array([
            rho_E_minus_small_t_NORM_alpha_time_varying_fn(
                t - t_E_aff,
                -1,
                ABL,
                ILD,
                params["rate_lambda"],
                params["T_0"],
                params["theta_E"],
                Z_E,
                np.nan,
                np.nan,
                params["rate_norm_l"],
                params["alpha"],
                True,
                False,
                K_MAX,
            )
            for t in T_PTS_REACTIVE_ONLY
        ])
        up_area = trapezoid(up_density, T_PTS_REACTIVE_ONLY)
        down_area = trapezoid(down_density, T_PTS_REACTIVE_ONLY)
        right_choice_probs.append(up_area / (up_area + down_area))

    return np.array(right_choice_probs)


# %%
with open(NPL_PSY_PKL, "rb") as handle:
    npl_psychometric_data = pickle.load(handle)
with open(ILD2_PSY_PKL, "rb") as handle:
    ild2_psychometric_data = pickle.load(handle)

npl_params = get_posterior_mean_params(NPL_RESULT_PKL, "vbmc_norm_tied_results")
ild2_params = get_posterior_mean_params(ILD2_RESULT_PKL, "vbmc_norm_alpha_ild2_delay_tied_results")
npl_avg_slope = average_sigmoid_slope(npl_psychometric_data)
ild2_avg_slope = average_sigmoid_slope(ild2_psychometric_data)


# %%
model_specs = {
    "NPL": {
        "color": "tab:blue",
        "psychometric_data": npl_psychometric_data,
        "formula_params": npl_params,
    },
    "NPL + alpha + ILD2": {
        "color": "tab:red",
        "psychometric_data": ild2_psychometric_data,
        "formula_params": ild2_params,
    },
}
abl_linestyles = {20: "-", 40: "--", 60: ":"}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

for model_label, spec in model_specs.items():
    model_data = spec["psychometric_data"][ANIMAL_KEY]
    for ABL in ABL_ARR:
        fit = model_data[ABL].get("fit")
        if fit is None or "params" not in fit:
            continue
        axes[0].plot(
            ILD_CONTINUOUS,
            sigmoid_from_fit_params(ILD_CONTINUOUS, fit["params"]),
            color=spec["color"],
            linestyle=abl_linestyles[ABL],
            linewidth=2,
            label=f"{model_label}, ABL={ABL}",
        )

    theta = spec["formula_params"]["theta_E"]
    rate_lambda = spec["formula_params"]["rate_lambda"]
    rate_norm_l = spec["formula_params"]["rate_norm_l"]
    alpha = spec["formula_params"]["alpha"]
    axes[1].plot(
        ILD_CONTINUOUS,
        gamma_alpha_formula_psychometric(ILD_CONTINUOUS, theta, rate_lambda, rate_norm_l, alpha),
        color=spec["color"],
        linewidth=2.5,
        label=f"{model_label}: theta={theta:.3f}, lambda={rate_lambda:.3f}, l={rate_norm_l:.3f}, alpha={alpha:.3f}",
    )

    for ABL in ABL_ARR:
        axes[2].plot(
            ILD_CONTINUOUS,
            reactive_only_psychometric(ILD_CONTINUOUS, ABL, spec["formula_params"]),
            color=spec["color"],
            linestyle=abl_linestyles[ABL],
            linewidth=2,
        )

axes[0].set_title(
    f"{BATCH_NAME}-{ANIMAL_ID}: Psychometric Fits\n"
    f"mean fitted sigmoid slope: NPL={npl_avg_slope:.3f}, ILD2={ild2_avg_slope:.3f}",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
)
axes[0].set_xlabel("ILD (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[0].set_ylabel("P(choice = right)", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[0].set_xticks([-16, -8, 0, 8, 16])
axes[0].set_yticks([0, 0.5, 1])
axes[0].set_ylim(-0.05, 1.05)
axes[0].axvline(0, color="0.6", linestyle="--", linewidth=1)
axes[0].axhline(0.5, color="0.6", linestyle="--", linewidth=1)
# axes[0].legend(frameon=False, fontsize=9, loc="lower right")

axes[1].set_title(r"$1/(1+\exp[-2\gamma_\alpha(ILD)])$", fontsize=ft.STYLE.LEGEND_FONTSIZE)

axes[1].set_xlabel("ILD (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[1].set_ylabel("Formula psychometric", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[1].set_xticks([-16, -8, 0, 8, 16])
axes[1].set_yticks([0, 0.5, 1])
axes[1].set_ylim(-0.05, 1.05)
axes[1].axvline(0, color="0.6", linestyle="--", linewidth=1)
axes[1].axhline(0.5, color="0.6", linestyle="--", linewidth=1)
# axes[1].legend(frameon=False, fontsize=9, loc="lower right")

axes[2].set_title(
    f"Reactive-only area psychometric\nw={REACTIVE_ONLY_W:.1f}, no proactive process",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
)
axes[2].set_xlabel("ILD (dB)", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[2].set_ylabel("P(choice = right)", fontsize=ft.STYLE.LABEL_FONTSIZE)
axes[2].set_xticks([-16, -8, 0, 8, 16])
axes[2].set_yticks([0, 0.5, 1])
axes[2].set_ylim(-0.05, 1.05)
axes[2].axvline(0, color="0.6", linestyle="--", linewidth=1)
axes[2].axhline(0.5, color="0.6", linestyle="--", linewidth=1)

for ax in axes:
    ax.tick_params(axis="both", labelsize=ft.STYLE.TICK_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

fig.suptitle(
    "Blue: NPL    Red: NPL + alpha + ILD2 delay    Line style: ABL 20 solid, 40 dashed, 60 dotted",
    fontsize=ft.STYLE.LEGEND_FONTSIZE,
    y=1.02,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

print(f"Saved {OUTPUT_PNG}")
print(
    "NPL formula params: "
    f"theta={npl_params['theta_E']:.6f}, "
    f"lambda={npl_params['rate_lambda']:.6f}, "
    f"rate_norm_l={npl_params['rate_norm_l']:.6f}, "
    f"alpha={npl_params['alpha']:.6f}"
)
print(
    "ILD2 formula params: "
    f"theta={ild2_params['theta_E']:.6f}, "
    f"lambda={ild2_params['rate_lambda']:.6f}, "
    f"rate_norm_l={ild2_params['rate_norm_l']:.6f}, "
    f"alpha={ild2_params['alpha']:.6f}"
)
print(f"Average fitted sigmoid slopes: NPL={npl_avg_slope:.6f}, ILD2={ild2_avg_slope:.6f}")
print(f"Reactive-only panel uses w={REACTIVE_ONLY_W:.3f} and omits P_A/C_A proactive terms")

# %%
