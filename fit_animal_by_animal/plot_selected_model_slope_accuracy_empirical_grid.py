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
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics_part2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")
IPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_vanilla.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "selected_models_slope_accuracy_empirical_grid.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "selected_models_slope_accuracy_empirical_grid.pdf")

ABL_ARR = [20, 40, 60]
N_SHUFFLES = 100_000
N_BOOTSTRAPS = 100_000
RNG_SEED = 0
N_BINS = 60
MODEL_CONFIGS = {
    "NPL": {
        "color": "tab:blue",
        "pkl": NPL_THEORETICAL_PSY_PKL,
    },
    "NPL+alpha+ILD2": {
        "color": "tab:red",
        "pkl": ILD2_THEORETICAL_PSY_PKL,
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


def paired_label_swap_test(model_values, data_values, rng):
    model_values = np.asarray(model_values, dtype=float)
    data_values = np.asarray(data_values, dtype=float)
    finite = np.isfinite(model_values) & np.isfinite(data_values)
    diff_values = model_values[finite] - data_values[finite]

    true_stat = float(np.mean(diff_values))
    signs = rng.choice([-1.0, 1.0], size=(N_SHUFFLES, len(diff_values)))
    null_stats = np.mean(signs * diff_values, axis=1)
    p_two_sided = (np.sum(np.abs(null_stats) >= abs(true_stat)) + 1) / (N_SHUFFLES + 1)

    return {
        "n_animals": int(len(diff_values)),
        "data_mean": float(np.mean(data_values[finite])),
        "model_mean": float(np.mean(model_values[finite])),
        "diff_values": diff_values,
        "true_stat": true_stat,
        "null_stats": null_stats,
        "p_two_sided": float(p_two_sided),
    }


def paired_abs_error_bootstrap(model_values, data_values, rng):
    model_values = np.asarray(model_values, dtype=float)
    data_values = np.asarray(data_values, dtype=float)
    finite = np.isfinite(model_values) & np.isfinite(data_values)
    abs_errors = np.abs(model_values[finite] - data_values[finite])

    true_stat = float(np.mean(abs_errors))
    sample_indices = rng.integers(0, len(abs_errors), size=(N_BOOTSTRAPS, len(abs_errors)))
    bootstrap_stats = np.mean(abs_errors[sample_indices], axis=1)
    ci_low, ci_high = np.percentile(bootstrap_stats, [2.5, 97.5])

    return {
        "n_animals": int(len(abs_errors)),
        "abs_errors": abs_errors,
        "true_stat": true_stat,
        "bootstrap_stats": bootstrap_stats,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


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


def setup_shuffle_axis(ax, test_result, color, title, x_label):
    null_stats = test_result["null_stats"]
    true_stat = test_result["true_stat"]

    ax.hist(null_stats, bins=N_BINS, color="0.72", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.axvline(true_stat, color=color, linestyle="-", linewidth=2.5)
    ax.set_title(
        f"{title}\n"
        f"mean diff={true_stat:+.4f}, p={test_result['p_two_sided']:.4g}, n={test_result['n_animals']}",
        fontsize=9.5,
    )
    ax.set_xlabel(x_label, fontsize=9.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def setup_abs_error_axis(ax, test_result, color, title, x_label):
    bootstrap_stats = test_result["bootstrap_stats"]
    true_stat = test_result["true_stat"]

    ax.hist(bootstrap_stats, bins=N_BINS, color="0.72", edgecolor="white", linewidth=0.5)
    ax.axvline(true_stat, color=color, linestyle="-", linewidth=2.5)
    ax.axvline(test_result["ci_low"], color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.axvline(test_result["ci_high"], color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_title(
        f"{title}\n"
        f"MAE={true_stat:.4f}, 95% CI [{test_result['ci_low']:.4f}, {test_result['ci_high']:.4f}], "
        f"n={test_result['n_animals']}",
        fontsize=9.5,
    )
    ax.set_xlabel(x_label, fontsize=9.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
rng = np.random.default_rng(RNG_SEED)
slope_values = []
accuracy_values = []
for entry in model_data.values():
    summary = entry["summary"]
    summary["slope_test"] = paired_label_swap_test(summary["model_slopes"], summary["data_slopes"], rng)
    summary["accuracy_test"] = paired_label_swap_test(summary["model_accuracy"], summary["data_accuracy"], rng)
    summary["slope_abs_error"] = paired_abs_error_bootstrap(summary["model_slopes"], summary["data_slopes"], rng)
    summary["accuracy_abs_error"] = paired_abs_error_bootstrap(summary["model_accuracy"], summary["data_accuracy"], rng)
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

fig, axes = plt.subplots(6, 3, figsize=(12.0, 20.5))

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
    setup_shuffle_axis(
        axes[2, col_idx],
        summary["slope_test"],
        color,
        f"{model_label}: slope label-swap",
        "Shuffled mean slope diff",
    )
    setup_shuffle_axis(
        axes[3, col_idx],
        summary["accuracy_test"],
        color,
        f"{model_label}: accuracy label-swap",
        "Shuffled mean accuracy diff",
    )
    setup_abs_error_axis(
        axes[4, col_idx],
        summary["slope_abs_error"],
        color,
        f"{model_label}: slope abs error",
        "Bootstrapped mean |slope diff|",
    )
    setup_abs_error_axis(
        axes[5, col_idx],
        summary["accuracy_abs_error"],
        color,
        f"{model_label}: accuracy abs error",
        "Bootstrapped mean |accuracy diff|",
    )

fig.suptitle(
    "Model vs Data on Empirical Stimulus Grids\n"
    "Rows 3-4 test signed bias; rows 5-6 show bootstrapped mean absolute error",
    fontsize=12,
    y=0.99,
)
axes[2, 0].set_ylabel("Shuffle count", fontsize=9.5)
axes[3, 0].set_ylabel("Shuffle count", fontsize=9.5)
axes[4, 0].set_ylabel("Bootstrap count", fontsize=9.5)
axes[5, 0].set_ylabel("Bootstrap count", fontsize=9.5)
fig.subplots_adjust(left=0.07, right=0.995, bottom=0.04, top=0.94, wspace=0.38, hspace=0.82)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

for model_label, entry in model_data.items():
    summary = entry["summary"]
    print(f"\n{model_label}")
    print(f"  slope: {metric_text(summary['data_slopes'], summary['model_slopes'])}")
    print(f"  accuracy: {metric_text(summary['data_accuracy'], summary['model_accuracy'])}")
    print(
        "  slope label-swap: "
        f"mean(model-data)={summary['slope_test']['true_stat']:+.6f}, "
        f"p={summary['slope_test']['p_two_sided']:.6f}, "
        f"n={summary['slope_test']['n_animals']}"
    )
    print(
        "  accuracy label-swap: "
        f"mean(model-data)={summary['accuracy_test']['true_stat']:+.6f}, "
        f"p={summary['accuracy_test']['p_two_sided']:.6f}, "
        f"n={summary['accuracy_test']['n_animals']}"
    )
    print(
        "  slope abs error bootstrap: "
        f"MAE={summary['slope_abs_error']['true_stat']:.6f}, "
        f"95% CI=[{summary['slope_abs_error']['ci_low']:.6f}, "
        f"{summary['slope_abs_error']['ci_high']:.6f}], "
        f"n={summary['slope_abs_error']['n_animals']}"
    )
    print(
        "  accuracy abs error bootstrap: "
        f"MAE={summary['accuracy_abs_error']['true_stat']:.6f}, "
        f"95% CI=[{summary['accuracy_abs_error']['ci_low']:.6f}, "
        f"{summary['accuracy_abs_error']['ci_high']:.6f}], "
        f"n={summary['accuracy_abs_error']['n_animals']}"
    )

# %%
