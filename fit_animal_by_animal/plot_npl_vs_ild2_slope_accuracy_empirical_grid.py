# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")

EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")
IPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_vanilla.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")
ILD2_WITH_NPL_DELAY_THEORETICAL_PSY_PKL = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "theoretical_psychometric_data_npl_alpha_ild2_with_npl_delay.pkl",
)
ILD2_WITH_NPL_DELAY_DELGO_THEORETICAL_PSY_PKL = os.path.join(
    ILD2_DIAGNOSTIC_DIR,
    "theoretical_psychometric_data_npl_alpha_ild2_with_npl_delay_delgo.pkl",
)

OUTPUT_PNG = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_vs_ild2_slope_accuracy_empirical_grid_with_npl_delay_delgo.png")
OUTPUT_PDF = os.path.join(ILD2_DIAGNOSTIC_DIR, "npl_vs_ild2_slope_accuracy_empirical_grid_with_npl_delay_delgo.pdf")

ABL_ARR = [20, 40, 60]
MODEL_CONFIGS = {
    "NPL": {
        "color": "tab:blue",
        "pkl": NPL_THEORETICAL_PSY_PKL,
    },
    "NPL+alpha+ILD2": {
        "color": "tab:red",
        "pkl": ILD2_THEORETICAL_PSY_PKL,
    },
    "ILD2 + NPL delay": {
        "color": "tab:purple",
        "pkl": ILD2_WITH_NPL_DELAY_THEORETICAL_PSY_PKL,
    },
    "ILD2 + NPL delay/go": {
        "color": "tab:brown",
        "pkl": ILD2_WITH_NPL_DELAY_DELGO_THEORETICAL_PSY_PKL,
    },
    "IPL / vanilla TIED": {
        "color": "tab:green",
        "pkl": IPL_THEORETICAL_PSY_PKL,
    },
}


# %%
def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    def sigmoid(x, upper, lower, x0, k):
        return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))

    ild_values = np.asarray(ild_values, dtype=float)
    right_choice_probs = np.asarray(right_choice_probs, dtype=float)
    valid_idx = np.isfinite(ild_values) & np.isfinite(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return np.nan

    try:
        popt, _ = curve_fit(
            sigmoid,
            ild_values[valid_idx],
            right_choice_probs[valid_idx],
            p0=[1.0, 0.0, 0.0, 1.0],
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]),
        )
        return float(popt[3])
    except Exception as exc:
        print(f"Could not fit sigmoid: {exc}")
        return np.nan


def get_right_prob_at_ild(condition_data, ILD):
    ild_values = np.asarray(condition_data["ild_values"], dtype=float)
    right_choice_probs = np.asarray(condition_data["right_choice_probs"], dtype=float)
    matches = np.where(np.isclose(ild_values, ILD))[0]
    if len(matches) == 0:
        return np.nan
    return float(right_choice_probs[matches[0]])


def signed_accuracy_from_right_prob(ILD, right_prob):
    if not np.isfinite(right_prob):
        return np.nan
    if ILD > 0:
        return right_prob
    if ILD < 0:
        return 1 - right_prob
    return np.nan


def slope_for_condition_on_its_ilds(condition_data):
    return fit_psychometric_sigmoid(
        condition_data["ild_values"],
        condition_data["right_choice_probs"],
    )


def model_slope_on_empirical_ilds(model_condition, empirical_condition):
    empirical_ilds = np.asarray(empirical_condition["ild_values"], dtype=float)
    model_probs = np.asarray([get_right_prob_at_ild(model_condition, ILD) for ILD in empirical_ilds], dtype=float)
    return fit_psychometric_sigmoid(empirical_ilds, model_probs)


def mean_accuracy_on_empirical_ilds(psy_data, animal_key, data_key, empirical_psy_data):
    accuracies = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        model_condition = psy_data.get(animal_key, {}).get(ABL, {}).get(data_key, None)
        if empirical_condition is None or model_condition is None:
            continue

        for ILD in np.asarray(empirical_condition["ild_values"], dtype=float):
            right_prob = get_right_prob_at_ild(model_condition, ILD)
            accuracies.append(signed_accuracy_from_right_prob(ILD, right_prob))

    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.size == 0 or np.all(~np.isfinite(accuracies)):
        return np.nan
    return float(np.nanmean(accuracies))


def get_slope_accuracy_data(model_psy_data, empirical_psy_data, animal_keys):
    data_slopes = []
    model_slopes = []
    data_accuracy = []
    model_accuracy = []

    for animal_key in animal_keys:
        animal_data_slopes = []
        animal_model_slopes = []
        for ABL in ABL_ARR:
            empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
            model_condition = model_psy_data.get(animal_key, {}).get(ABL, {}).get("theoretical", None)
            if empirical_condition is None or model_condition is None:
                continue

            animal_data_slopes.append(slope_for_condition_on_its_ilds(empirical_condition))
            animal_model_slopes.append(model_slope_on_empirical_ilds(model_condition, empirical_condition))

        data_slopes.append(np.nanmean(animal_data_slopes))
        model_slopes.append(np.nanmean(animal_model_slopes))
        data_accuracy.append(mean_accuracy_on_empirical_ilds(empirical_psy_data, animal_key, "empirical", empirical_psy_data))
        model_accuracy.append(mean_accuracy_on_empirical_ilds(model_psy_data, animal_key, "theoretical", empirical_psy_data))

    return {
        "data_slopes": np.asarray(data_slopes, dtype=float),
        "model_slopes": np.asarray(model_slopes, dtype=float),
        "data_accuracy": np.asarray(data_accuracy, dtype=float),
        "model_accuracy": np.asarray(model_accuracy, dtype=float),
    }


def metric_text(x_values, y_values):
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.sum(finite) < 2:
        return "R²=nan, MSE=nan"
    return (
        f"R²={r2_score(x_values[finite], y_values[finite]):.3f}, "
        f"MSE={mean_squared_error(x_values[finite], y_values[finite]):.5f}"
    )


def setup_scatter_axis(ax, x_values, y_values, color, title, x_label, y_label, axis_limits, axis_ticks):
    ax.scatter(
        x_values,
        y_values,
        marker="o",
        s=70,
        color=color,
        alpha=0.5,
        edgecolors="none",
    )
    ax.plot(axis_limits, axis_limits, color="grey", linestyle="--", linewidth=1.6, alpha=0.55, zorder=0)
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(f"{title}\n{metric_text(x_values, y_values)}", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


# %%
with open(EMPIRICAL_PSY_PKL, "rb") as handle:
    empirical_psy_data = pickle.load(handle)

model_data = {}
common_animal_sets = [set(empirical_psy_data)]
for model_label, config in MODEL_CONFIGS.items():
    with open(config["pkl"], "rb") as handle:
        model_psy_data = pickle.load(handle)
    model_data[model_label] = {
        "psy_data": model_psy_data,
        "color": config["color"],
    }
    common_animal_sets.append(set(model_psy_data))

animal_keys = sorted(set.intersection(*common_animal_sets), key=lambda item: (item[0], item[1]))
print(f"Using {len(animal_keys)} animals")
print(f"SD animals included: {[animal_key for animal_key in animal_keys if animal_key[0] == 'SD']}")

for model_label, entry in model_data.items():
    entry["summary"] = get_slope_accuracy_data(entry["psy_data"], empirical_psy_data, animal_keys)


# %%
slope_values = []
accuracy_values = []
for entry in model_data.values():
    summary = entry["summary"]
    slope_values.extend(summary["data_slopes"])
    slope_values.extend(summary["model_slopes"])
    accuracy_values.extend(summary["data_accuracy"])
    accuracy_values.extend(summary["model_accuracy"])

slope_values = np.asarray(slope_values, dtype=float)
accuracy_values = np.asarray(accuracy_values, dtype=float)

slope_axis_limits = [
    max(0.0, np.floor((np.nanmin(slope_values) - 0.04) * 10) / 10),
    min(1.0, np.ceil((np.nanmax(slope_values) + 0.04) * 10) / 10),
]
slope_axis_ticks = [
    slope_axis_limits[0],
    (slope_axis_limits[0] + slope_axis_limits[1]) / 2,
    slope_axis_limits[1],
]
accuracy_axis_limits = [0.7, 0.9]
accuracy_axis_ticks = [0.7, 0.8, 0.9]

fig, axes = plt.subplots(2, 5, figsize=(18.8, 8.6))

for col_idx, (model_label, entry) in enumerate(model_data.items()):
    summary = entry["summary"]
    color = entry["color"]
    setup_scatter_axis(
        axes[0, col_idx],
        summary["data_slopes"],
        summary["model_slopes"],
        color,
        f"{model_label}: psychometric slope",
        "Data slope",
        "Model slope",
        slope_axis_limits,
        slope_axis_ticks,
    )
    setup_scatter_axis(
        axes[1, col_idx],
        summary["data_accuracy"],
        summary["model_accuracy"],
        color,
        f"{model_label}: accuracy",
        "Data accuracy",
        "Model accuracy",
        accuracy_axis_limits,
        accuracy_axis_ticks,
    )

fig.suptitle(
    "Model vs Data on Empirical Stimulus Grids\n"
    "For SD animals, model slope and accuracy use only ILDs present in data",
    fontsize=12,
    y=0.98,
)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.08, top=0.85, wspace=0.36, hspace=0.58)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

for model_label, entry in model_data.items():
    summary = entry["summary"]
    print(f"\n{model_label}")
    print(f"  slope: {metric_text(summary['data_slopes'], summary['model_slopes'])}")
    print(f"  accuracy: {metric_text(summary['data_accuracy'], summary['model_accuracy'])}")

# %%
