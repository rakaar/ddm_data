# %%
import csv
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


# %%
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

ILD2_DIAGNOSTIC_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "figure_4_diagnostics")
ILD2_RESULTS_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "result_pkls")
OUTPUT_DIR = os.path.join(REPO_DIR, "NPL_alpha_ILD2_fit_results", "param_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMPIRICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "empirical_psychometric_data_for_npl_alpha_ild2.pkl")
NPL_THEORETICAL_PSY_PKL = os.path.join(SCRIPT_DIR, "theoretical_psychometric_data_norm.pkl")
ILD2_THEORETICAL_PSY_PKL = os.path.join(ILD2_DIAGNOSTIC_DIR, "theoretical_psychometric_data_npl_alpha_ild2.pkl")

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "ild2_accuracy_change_vs_params.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "ild2_accuracy_change_vs_params.pdf")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ild2_accuracy_change_vs_params.csv")
FOCUSED_OUTPUT_PNG = os.path.join(OUTPUT_DIR, "accuracy_percent_change_vs_param_percent_change.png")
FOCUSED_OUTPUT_PDF = os.path.join(OUTPUT_DIR, "accuracy_percent_change_vs_param_percent_change.pdf")

NPL_RESULT_KEY = "vbmc_norm_tied_results"
ILD2_RESULT_KEY = "vbmc_norm_alpha_ild2_delay_tied_results"

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
BATCH_ORDER = {batch: idx for idx, batch in enumerate(DESIRED_BATCHES)}
ABL_ARR = [20, 40, 60]

COMMON_PARAM_CONFIGS = [
    ("rate_lambda_samples", "rate_lambda", r"$\lambda$", 1.0),
    ("T_0_samples", "T_0_ms", r"$T_0$ (ms)", 1e3),
    ("theta_E_samples", "theta_E", r"$\theta_E$", 1.0),
    ("w_samples", "w", r"$w$", 1.0),
    ("del_go_samples", "del_go", r"$\Delta_{go}$", 1.0),
    ("rate_norm_l_samples", "rate_norm_l", r"$\ell$", 1.0),
]

ILD2_ONLY_PARAM_CONFIGS = [
    ("alpha_samples", "alpha", r"$\alpha$", 1.0),
    ("bias_ms_samples", "delay_bias_ms", "delay bias (ms)", 1.0),
    ("abl_delay_coeff_ms_per_abl_samples", "delay_ABL_coeff", "ABL coeff", 1.0),
    ("abs_ild_delay_coeff_ms_per_unit_samples", "delay_absILD_coeff", "|ILD| coeff", 1.0),
    ("ild2_delay_coeff_ms_per_unit2_samples", "delay_ILD2_coeff", r"ILD$^2$ coeff", 1.0),
]


# %%
def sample_mean(results, result_key, sample_key, scale, pkl_path):
    if result_key not in results:
        raise KeyError(f"{pkl_path} is missing `{result_key}`")
    if sample_key not in results[result_key]:
        raise KeyError(f"{pkl_path} is missing `{result_key}[{sample_key}]`")
    samples = np.asarray(results[result_key][sample_key], dtype=float) * scale
    if samples.size == 0:
        raise ValueError(f"{pkl_path} has empty samples for `{result_key}[{sample_key}]`")
    return float(np.nanmean(samples))


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


def mean_model_accuracy_on_empirical_conditions(model_psy_data, animal_key, empirical_psy_data):
    accuracies = []
    empirical_ilds_all = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        model_condition = model_psy_data.get(animal_key, {}).get(ABL, {}).get("theoretical", None)
        if empirical_condition is None or model_condition is None:
            continue

        empirical_ilds = np.asarray(empirical_condition["ild_values"], dtype=float)
        empirical_ilds_all.extend(empirical_ilds)
        for ILD in empirical_ilds:
            right_prob = get_right_prob_at_ild(model_condition, ILD)
            accuracies.append(signed_accuracy_from_right_prob(ILD, right_prob))

    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.size == 0 or np.all(~np.isfinite(accuracies)):
        return np.nan, np.asarray(empirical_ilds_all, dtype=float)
    return float(np.nanmean(accuracies)), np.asarray(empirical_ilds_all, dtype=float)


def mean_ild2_delay_on_empirical_conditions(ild2_params, animal_key, empirical_psy_data):
    delays = []
    for ABL in ABL_ARR:
        empirical_condition = empirical_psy_data.get(animal_key, {}).get(ABL, {}).get("empirical", None)
        if empirical_condition is None:
            continue

        for ILD in np.asarray(empirical_condition["ild_values"], dtype=float):
            delay_ms = (
                ild2_params["delay_bias_ms"]
                + ild2_params["delay_ABL_coeff"] * ABL
                + ild2_params["delay_absILD_coeff"] * abs(ILD)
                + ild2_params["delay_ILD2_coeff"] * ILD**2
            )
            delays.append(delay_ms)

    delays = np.asarray(delays, dtype=float)
    if delays.size == 0 or np.all(~np.isfinite(delays)):
        return np.nan
    return float(np.nanmean(delays))


def corr_text(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.sum(finite) < 3:
        return "r=nan"
    return f"r={np.corrcoef(x_values[finite], y_values[finite])[0, 1]:+.2f}"


def scatter_param_vs_accuracy_change(ax, rows, param_name, param_label, title=None, annotate_extremes=False):
    x_values = np.asarray([row[param_name] for row in rows], dtype=float)
    y_values = np.asarray([row["ild2_vs_npl_accuracy_percent_change"] for row in rows], dtype=float)
    colors = np.where(y_values < 0, "tab:red", "tab:green")
    ax.scatter(x_values, y_values, s=58, color=colors, alpha=0.75, edgecolors="none")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.set_xlabel(param_label)
    ax.set_ylabel("ILD2 vs NPL accuracy change (%)")
    ax.set_title(f"{title or param_label}\n{corr_text(x_values, y_values)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_extremes:
        y_order = np.argsort(y_values)
        label_indices = list(y_order[:3]) + list(y_order[-3:])
        for idx in label_indices:
            if not (np.isfinite(x_values[idx]) and np.isfinite(y_values[idx])):
                continue
            ax.annotate(
                rows[idx]["label"],
                xy=(x_values[idx], y_values[idx]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )


def scatter_with_fit_line(ax, rows, param_name, param_label, title):
    x_values = np.asarray([row[param_name] for row in rows], dtype=float)
    y_values = np.asarray([row["ild2_vs_npl_accuracy_percent_change"] for row in rows], dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    colors = np.where(y_values < 0, "tab:red", "tab:green")

    ax.scatter(
        x_values,
        y_values,
        s=64,
        color=colors,
        alpha=0.72,
        edgecolors="none",
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=1.1, alpha=0.65)

    if np.sum(finite) >= 3:
        slope, intercept = np.polyfit(x_values[finite], y_values[finite], 1)
        line_x = np.linspace(np.nanmin(x_values[finite]), np.nanmax(x_values[finite]), 100)
        ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.4, alpha=0.75)

    ax.set_xlabel(param_label)
    ax.set_ylabel("accuracy change (%)")
    ax.set_title(f"{title}\n{corr_text(x_values, y_values)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# %%
npl_pattern = re.compile(r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)\.pkl$")
ild2_pattern = re.compile(
    r"^results_(?P<batch>.+)_animal_(?P<animal>\d+)_NORM_ALPHA_ILD2_DELAY_FROM_ABORTS\.pkl$"
)

npl_paths = {}
for fname in os.listdir(SCRIPT_DIR):
    match = npl_pattern.match(fname)
    if match is None:
        continue
    npl_paths[(match.group("batch"), int(match.group("animal")))] = os.path.join(SCRIPT_DIR, fname)

ild2_paths = {}
for fname in os.listdir(ILD2_RESULTS_DIR):
    match = ild2_pattern.match(fname)
    if match is None:
        continue
    ild2_paths[(match.group("batch"), int(match.group("animal")))] = os.path.join(ILD2_RESULTS_DIR, fname)

with open(EMPIRICAL_PSY_PKL, "rb") as handle:
    empirical_psy_data = pickle.load(handle)
with open(NPL_THEORETICAL_PSY_PKL, "rb") as handle:
    npl_theoretical_psy_data = pickle.load(handle)
with open(ILD2_THEORETICAL_PSY_PKL, "rb") as handle:
    ild2_theoretical_psy_data = pickle.load(handle)

matched_pairs = sorted(
    set(npl_paths) & set(ild2_paths) & set(empirical_psy_data) & set(npl_theoretical_psy_data) & set(ild2_theoretical_psy_data),
    key=lambda pair: (BATCH_ORDER.get(pair[0], len(BATCH_ORDER)), pair[1]),
)

if len(matched_pairs) == 0:
    raise RuntimeError("No matched NPL/NPL+alpha+ILD2 animals found.")

print(f"Using {len(matched_pairs)} matched animals")

rows = []
for animal_key in matched_pairs:
    batch, animal = animal_key
    npl_path = npl_paths[animal_key]
    ild2_path = ild2_paths[animal_key]

    with open(npl_path, "rb") as handle:
        npl_results = pickle.load(handle)
    with open(ild2_path, "rb") as handle:
        ild2_results = pickle.load(handle)

    npl_accuracy, empirical_ilds = mean_model_accuracy_on_empirical_conditions(
        npl_theoretical_psy_data,
        animal_key,
        empirical_psy_data,
    )
    ild2_accuracy, _ = mean_model_accuracy_on_empirical_conditions(
        ild2_theoretical_psy_data,
        animal_key,
        empirical_psy_data,
    )

    row = {
        "batch": batch,
        "animal": animal,
        "label": f"{batch}-{animal}",
        "npl_accuracy": npl_accuracy,
        "ild2_accuracy": ild2_accuracy,
        "ild2_minus_npl_accuracy": ild2_accuracy - npl_accuracy,
        "ild2_vs_npl_accuracy_percent_change": 100 * (ild2_accuracy - npl_accuracy) / npl_accuracy,
    }

    for sample_key, param_name, _, scale in COMMON_PARAM_CONFIGS:
        npl_mean = sample_mean(npl_results, NPL_RESULT_KEY, sample_key, scale, npl_path)
        ild2_mean = sample_mean(ild2_results, ILD2_RESULT_KEY, sample_key, scale, ild2_path)
        row[f"npl_{param_name}"] = npl_mean
        row[f"ild2_{param_name}"] = ild2_mean
        row[f"delta_{param_name}"] = ild2_mean - npl_mean
        row[f"pct_delta_{param_name}"] = 100 * (ild2_mean - npl_mean) / npl_mean

    for sample_key, param_name, _, scale in ILD2_ONLY_PARAM_CONFIGS:
        row[param_name] = sample_mean(ild2_results, ILD2_RESULT_KEY, sample_key, scale, ild2_path)

    row["mean_ild2_delay_ms_empirical_grid"] = mean_ild2_delay_on_empirical_conditions(row, animal_key, empirical_psy_data)
    row["empirical_abs_ild_max"] = float(np.nanmax(np.abs(empirical_ilds))) if empirical_ilds.size else np.nan
    rows.append(row)


# %%
fieldnames = list(rows[0].keys())
with open(OUTPUT_CSV, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {OUTPUT_CSV}")

increase_rows = [row for row in rows if row["ild2_vs_npl_accuracy_percent_change"] > 0]
decrease_rows = [row for row in rows if row["ild2_vs_npl_accuracy_percent_change"] < 0]

print("\nGroup summaries: increase vs decrease")
for param_name in ["alpha", "ild2_rate_norm_l", "delay_bias_ms", "delay_ILD2_coeff", "mean_ild2_delay_ms_empirical_grid"]:
    inc_vals = np.asarray([row[param_name] for row in increase_rows], dtype=float)
    dec_vals = np.asarray([row[param_name] for row in decrease_rows], dtype=float)
    print(
        f"  {param_name}: "
        f"increase mean={np.nanmean(inc_vals):.4f}, median={np.nanmedian(inc_vals):.4f}; "
        f"decrease mean={np.nanmean(dec_vals):.4f}, median={np.nanmedian(dec_vals):.4f}"
    )

print("\nCorrelations with ILD2-vs-NPL accuracy percent change:")
for param_name in [
    "alpha",
    "ild2_rate_norm_l",
    "delay_bias_ms",
    "delay_ABL_coeff",
    "delay_absILD_coeff",
    "delay_ILD2_coeff",
    "mean_ild2_delay_ms_empirical_grid",
    "pct_delta_rate_lambda",
    "pct_delta_T_0_ms",
    "pct_delta_theta_E",
    "pct_delta_w",
    "pct_delta_rate_norm_l",
]:
    print(f"  {param_name}: {corr_text([row[param_name] for row in rows], [row['ild2_vs_npl_accuracy_percent_change'] for row in rows])}")

print("\nLargest ILD2 accuracy increases:")
for row in sorted(rows, key=lambda item: item["ild2_vs_npl_accuracy_percent_change"], reverse=True)[:8]:
    print(
        f"  {row['label']}: pct={row['ild2_vs_npl_accuracy_percent_change']:+.2f}%, "
        f"alpha={row['alpha']:.3f}, ell={row['ild2_rate_norm_l']:.3f}, "
        f"mean_delay={row['mean_ild2_delay_ms_empirical_grid']:.1f} ms"
    )

print("\nLargest ILD2 accuracy decreases:")
for row in sorted(rows, key=lambda item: item["ild2_vs_npl_accuracy_percent_change"])[:8]:
    print(
        f"  {row['label']}: pct={row['ild2_vs_npl_accuracy_percent_change']:+.2f}%, "
        f"alpha={row['alpha']:.3f}, ell={row['ild2_rate_norm_l']:.3f}, "
        f"mean_delay={row['mean_ild2_delay_ms_empirical_grid']:.1f} ms"
    )


# %%
fig, axes = plt.subplots(3, 4, figsize=(13.0, 10.0))
plot_specs = [
    ("alpha", r"$\alpha$", "ILD2 alpha", True),
    ("ild2_rate_norm_l", r"ILD2 $\ell$", "ILD2 norm exponent", False),
    ("mean_ild2_delay_ms_empirical_grid", "mean delay (ms)", "ILD2 empirical-grid delay", False),
    ("delay_ILD2_coeff", r"ILD$^2$ delay coeff", r"ILD$^2$ delay coeff", False),
    ("pct_delta_rate_lambda", r"% $\Delta\lambda$", r"ILD2 vs NPL $\lambda$", False),
    ("pct_delta_T_0_ms", r"% $\Delta T_0$", r"ILD2 vs NPL $T_0$", False),
    ("pct_delta_theta_E", r"% $\Delta\theta_E$", r"ILD2 vs NPL $\theta_E$", False),
    ("pct_delta_w", r"% $\Delta w$", r"ILD2 vs NPL $w$", False),
    ("pct_delta_del_go", r"% $\Delta\Delta_{go}$", r"ILD2 vs NPL $\Delta_{go}$", False),
    ("pct_delta_rate_norm_l", r"% $\Delta\ell$", r"ILD2 vs NPL $\ell$", False),
    ("delay_ABL_coeff", "ABL delay coeff", "delay ABL coeff", False),
    ("delay_absILD_coeff", "|ILD| delay coeff", "delay |ILD| coeff", False),
]

for ax, (param_name, param_label, title, annotate_extremes) in zip(axes.ravel(), plot_specs):
    scatter_param_vs_accuracy_change(
        ax,
        rows,
        param_name,
        param_label,
        title=title,
        annotate_extremes=annotate_extremes,
    )

fig.suptitle(
    "NPL + alpha + ILD2 Accuracy Change vs Parameters\n"
    "Accuracy change = 100 x (ILD2 accuracy - NPL accuracy) / NPL accuracy; SD evaluated only on empirical |ILD| = 1, 2, 4, 8",
    fontsize=13,
    y=0.985,
)
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.89, wspace=0.40, hspace=0.62)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"\nSaved {OUTPUT_PNG}")
print(f"Saved {OUTPUT_PDF}")

# %%
fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.6))
focused_plot_specs = [
    ("alpha", r"$\alpha$", r"Accuracy change vs $\alpha$"),
    ("pct_delta_rate_lambda", r"% $\Delta\lambda$", r"Accuracy vs % $\Delta\lambda$"),
    ("pct_delta_theta_E", r"% $\Delta\theta_E$", r"Accuracy vs % $\Delta\theta_E$"),
    ("pct_delta_T_0_ms", r"% $\Delta T_0$", r"Accuracy vs % $\Delta T_0$"),
    ("pct_delta_rate_norm_l", r"% $\Delta\ell$", r"Accuracy vs % $\Delta\ell$"),
]

print("\nFocused parameter-change correlations:")
for ax, (param_name, param_label, title) in zip(axes, focused_plot_specs):
    scatter_with_fit_line(ax, rows, param_name, param_label, title)
    print(
        f"  {param_name}: "
        f"{corr_text([row[param_name] for row in rows], [row['ild2_vs_npl_accuracy_percent_change'] for row in rows])}"
    )

fig.suptitle(
    "NPL + alpha + ILD2 Accuracy Change vs Parameter Change\n"
    r"For $\lambda$, $\theta_E$, $T_0$, and $\ell$: x = 100 x (ILD2 parameter - NPL parameter) / NPL parameter",
    fontsize=12,
    y=1.04,
)
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.72, wspace=0.45)
fig.savefig(FOCUSED_OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FOCUSED_OUTPUT_PDF, dpi=300, bbox_inches="tight")

print(f"\nSaved {FOCUSED_OUTPUT_PNG}")
print(f"Saved {FOCUSED_OUTPUT_PDF}")

# %%
