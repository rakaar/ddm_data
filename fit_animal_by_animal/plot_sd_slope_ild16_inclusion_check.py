# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")

MODEL_SOURCES = {
    "npl": {
        "label": "NPL",
        "theory_pkl": os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl"),
        "output_png": os.path.join(ILD2_DIAGNOSTIC_DIR, "sd_slope_ild16_inclusion_check_npl.png"),
        "output_pdf": os.path.join(ILD2_DIAGNOSTIC_DIR, "sd_slope_ild16_inclusion_check_npl.pdf"),
    },
    "npl_alpha_ild2": {
        "label": "NPL + alpha + ILD2 delay",
        "theory_pkl": os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl"),
        "output_png": os.path.join(ILD2_DIAGNOSTIC_DIR, "sd_slope_ild16_inclusion_check_npl_alpha_ild2.png"),
        "output_pdf": os.path.join(ILD2_DIAGNOSTIC_DIR, "sd_slope_ild16_inclusion_check_npl_alpha_ild2.pdf"),
    },
}

ABL_ARR = [20, 40, 60]


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


def slope_for_condition(condition_data):
    return fit_psychometric_sigmoid(
        condition_data["ild_values"],
        condition_data["right_choice_probs"],
    )


def model_slope_on_empirical_ilds(theory_condition, empirical_condition):
    empirical_ilds = np.asarray(empirical_condition["ild_values"], dtype=float)
    theory_probs = np.asarray([get_right_prob_at_ild(theory_condition, ILD) for ILD in empirical_ilds], dtype=float)
    return fit_psychometric_sigmoid(empirical_ilds, theory_probs)


# %%
with open(EMPIRICAL_PSY_PKL, "rb") as handle:
    empirical_psy_data = pickle.load(handle)

sd_animals = sorted([animal_key for animal_key in empirical_psy_data if animal_key[0] == "SD"], key=lambda item: item[1])
print(f"SD animals: {sd_animals}")


# %%
def get_sd_slope_summary(theoretical_psy_data):
    data_mean_slopes = []
    model_all_ild_mean_slopes = []
    model_empirical_ild_mean_slopes = []
    animal_labels = []

    for animal_key in sd_animals:
        data_slopes = []
        model_all_ild_slopes = []
        model_empirical_ild_slopes = []
        for ABL in ABL_ARR:
            empirical_condition = empirical_psy_data[animal_key][ABL]["empirical"]
            theory_condition = theoretical_psy_data[animal_key][ABL]["theoretical"]

            data_slopes.append(slope_for_condition(empirical_condition))
            model_all_ild_slopes.append(slope_for_condition(theory_condition))
            model_empirical_ild_slopes.append(model_slope_on_empirical_ilds(theory_condition, empirical_condition))

        animal_labels.append(f"{animal_key[0]}-{animal_key[1]}")
        data_mean_slopes.append(np.nanmean(data_slopes))
        model_all_ild_mean_slopes.append(np.nanmean(model_all_ild_slopes))
        model_empirical_ild_mean_slopes.append(np.nanmean(model_empirical_ild_slopes))

    return {
        "animal_labels": animal_labels,
        "data": np.asarray(data_mean_slopes, dtype=float),
        "model_all_ild": np.asarray(model_all_ild_mean_slopes, dtype=float),
        "model_empirical_ild": np.asarray(model_empirical_ild_mean_slopes, dtype=float),
    }


def setup_slope_axis(ax, title, axis_limits, axis_ticks):
    ax.plot(axis_limits, axis_limits, color="grey", linestyle="--", linewidth=1.5, alpha=0.55, zorder=0)
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_xlabel("Data slope")
    ax.set_ylabel("Model slope")
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def annotate_animals(ax, x_values, y_values, animal_labels):
    for x_val, y_val, label in zip(x_values, y_values, animal_labels):
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        ax.annotate(label, (x_val, y_val), xytext=(4, 3), textcoords="offset points", fontsize=8, alpha=0.8)


def plot_sd_slope_check(model_label, summary, output_png, output_pdf):
    data_slopes = summary["data"]
    model_all_ild_slopes = summary["model_all_ild"]
    model_empirical_ild_slopes = summary["model_empirical_ild"]
    animal_labels = summary["animal_labels"]

    all_values = np.concatenate([data_slopes, model_all_ild_slopes, model_empirical_ild_slopes])
    finite_values = all_values[np.isfinite(all_values)]
    axis_min = max(0.0, np.floor((np.nanmin(finite_values) - 0.04) * 10) / 10)
    axis_max = min(1.0, np.ceil((np.nanmax(finite_values) + 0.04) * 10) / 10)
    axis_limits = [axis_min, axis_max]
    axis_ticks = [axis_min, (axis_min + axis_max) / 2, axis_max]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    axes[0].scatter(data_slopes, model_all_ild_slopes, s=70, color="tab:red", alpha=0.4, edgecolors="none")
    annotate_animals(axes[0], data_slopes, model_all_ild_slopes, animal_labels)
    setup_slope_axis(axes[0], "Model fit on all theoretical ILDs\n(includes +/-16)", axis_limits, axis_ticks)

    axes[1].scatter(data_slopes, model_empirical_ild_slopes, s=70, color="tab:green", alpha=0.4, edgecolors="none")
    annotate_animals(axes[1], data_slopes, model_empirical_ild_slopes, animal_labels)
    setup_slope_axis(axes[1], "Model refit on empirical ILDs only", axis_limits, axis_ticks)

    axes[2].scatter(
        data_slopes,
        model_all_ild_slopes,
        s=80,
        color="tab:red",
        alpha=0.4,
        edgecolors="none",
    )
    axes[2].scatter(
        data_slopes,
        model_empirical_ild_slopes,
        s=80,
        color="tab:green",
        alpha=0.4,
        edgecolors="none",
    )
    annotate_animals(axes[2], data_slopes, model_empirical_ild_slopes, animal_labels)
    setup_slope_axis(axes[2], "Overlap\nred: all ILDs, green: empirical only", axis_limits, axis_ticks)

    deltas = model_empirical_ild_slopes - model_all_ild_slopes
    fig.suptitle(
        f"SD Animals: Effect of Excluding Model-Only +/-16 ILDs on {model_label} Slope\n"
        f"Mean(model slope using empirical ILDs only - model slope using all ILDs) = {np.nanmean(deltas):+.4f}",
        fontsize=12,
        y=1.03,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.76, wspace=0.38)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")

    print(f"\n{model_label}: SD mean slopes")
    for label, data_slope, all_slope, empirical_slope, delta in zip(
        animal_labels,
        data_slopes,
        model_all_ild_slopes,
        model_empirical_ild_slopes,
        deltas,
    ):
        print(
            f"  {label}: data={data_slope:.4f}, "
            f"all_ILD_model={all_slope:.4f}, empirical_ILD_model={empirical_slope:.4f}, "
            f"delta={delta:+.4f}"
        )


# %%
for model_key, config in MODEL_SOURCES.items():
    with open(config["theory_pkl"], "rb") as handle:
        theoretical_psy_data = pickle.load(handle)

    summary = get_sd_slope_summary(theoretical_psy_data)
    plot_sd_slope_check(
        config["label"],
        summary,
        config["output_png"],
        config["output_pdf"],
    )

# %%
