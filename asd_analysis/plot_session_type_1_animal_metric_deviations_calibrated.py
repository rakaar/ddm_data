# %% Imports
"""Refit animal psychometrics after the CNTNAP2 cohort 4 calibration cleanup.

Two Figure 1-style PNG/PDF pairs retain the earlier arrangement and signed
animal-centered delta. Historical fits/figures are comparison inputs only.
Run with .venv/bin/python from the repository root.
"""
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_slope_jnd"
PREVIOUS_PLOTS = REPO / "asd_analysis/session_type_1_animal_metric_deviations"
OUTPUT = REPO / "asd_analysis/session_type_1_animal_metric_deviations_calibrated"
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
FIGSIZE = (26, 10)
PNG_DPI = 180
DOT_ALPHA = .65
POPULATION_SD_DDOF = 0
EXPECTED = {"animals": 53, "trials": 451381, "right": 231263, "left": 220118}
METRICS = {
    "jnd": {"source": "jnd", "title": "JND", "units": "dB", "animals": 49,
            "trials": 420616, "raw_tick": .5, "delta_tick": .2, "filename": "jnd_and_delta_jnd"},
    "k": {"source": "slope_k", "title": "Slope = k", "units": "dB⁻¹", "animals": 53,
          "trials": 451381, "raw_tick": .05, "delta_tick": .02, "filename": "slope_and_delta_slope"},
}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Preserve inputs and every earlier analysis/figure; establish before/after lineage
cleanup = json.loads((INPUT.parent / "speaker_calibration_cleanup.json").read_text())
reference_manifest = json.loads((REFERENCE / "slope_jnd_validation.json").read_text())
assert cleanup["status"] == reference_manifest["status"] == "passed"
lineage = cleanup["files"][str(INPUT.relative_to(REPO))]
assert lineage["before"]["sha256"] == reference_manifest["input_sha256"]
protected = set(REPO.glob("SHANK3_cohort*.csv")) | set(REPO.glob("CNTNAP2_cohort*.csv"))
protected.update(INPUT.parent.glob("*.csv"))
protected.update(INPUT.parent.glob("*.json"))
for folder in (REPO / "asd_analysis").glob("session_type_1_*"):
    if folder.is_dir() and folder != OUTPUT:
        protected.update(p for p in folder.rglob("*") if p.is_file())
for date in ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]:
    protected.update(p for p in (REPO / f"docs/assets/results/{date}").iterdir() if p.is_file())
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
assert input_sha256 == lineage["after"]["sha256"]


# %% Identical four-parameter fit and paper-defined inverse-sigmoid JND
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


def jnd_from_fit(upper, lower, x0, k):
    """Use the exact Figure 1 thresholds; audit undefined crossings explicitly."""
    high = .75 * upper
    low = .25 * (1 - lower)
    result = {"y_high": high, "y_low": low, "ild_high": np.nan,
              "ild_low": np.nan, "jnd": np.nan, "jnd_status": "invalid_fit"}
    if not np.isfinite([upper, lower, x0, k]).all() or upper <= lower or k <= 0:
        return result
    for side, threshold in [("high", high), ("low", low)]:
        if lower < threshold < upper:
            result[f"ild_{side}"] = x0 - np.log((upper - lower) / (threshold - lower) - 1) / k
    if not lower < low < high < upper:
        result["jnd_status"] = "thresholds_outside_asymptotes_or_reversed"
        return result
    value = (result["ild_high"] - result["ild_low"]) / 2
    if np.isfinite([result["ild_high"], result["ild_low"], value]).all() and value > 0:
        result.update(jnd=float(value), jnd_status="ok")
    else:
        result["jnd_status"] = "nonfinite_or_nonpositive_jnd"
    return result


def fit_condition(x, probability):
    """Match the paper's initial guess, bounds, and unweighted curve_fit objective."""
    initial = [float(np.max(probability)), float(np.min(probability)), float(np.median(x)), INITIAL_K]
    result = {"upper": np.nan, "lower": np.nan, "x0": np.nan, "k": np.nan,
              "slope_k": np.nan, "jnd": np.nan, "y_high": np.nan, "y_low": np.nan,
              "ild_high": np.nan, "ild_low": np.nan, "fit_status": "optimizer_failed",
              "jnd_status": "fit_unavailable", "rmse": np.nan, "unweighted_sse": np.nan,
              "optimizer_message": "", "optimizer_code": 0, "function_evaluations": 0,
              "warnings": "", "covariance_finite": False, "initial_parameters": json.dumps(initial)}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, code = curve_fit(
                sigmoid, x, probability, p0=initial, bounds=FIT_BOUNDS,
                maxfev=MAXFEV, full_output=True,
            )
        upper, lower, x0, k = parameters
        result.update(upper=upper, lower=lower, x0=x0, k=k,
                      optimizer_message=message, optimizer_code=int(code),
                      function_evaluations=int(info["nfev"]),
                      warnings="; ".join(dict.fromkeys(str(w.message) for w in caught)),
                      covariance_finite=bool(np.isfinite(covariance).all()))
        if not np.isfinite(parameters).all():
            result["fit_status"] = "nonfinite_parameters"
            return result
        residual = sigmoid(x, *parameters) - probability
        result.update(rmse=float(np.sqrt(np.mean(residual ** 2))),
                      unweighted_sse=float(np.dot(residual, residual)))
        if upper <= lower or k <= 0:
            result["fit_status"] = "non_increasing"
            return result
        result.update(fit_status="ok", slope_k=float(k))
        result.update(jnd_from_fit(*parameters))
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["optimizer_message"] = str(error)
    return result


# %% Current valid choices: no RT restriction and no repeated-trial refiltering
data = pd.read_csv(INPUT, usecols=[
    "filename", "animal", "genotype", "session", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success", "response_poke",
], low_memory=False)
assert len(data) == 610603 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
assert not (data.filename.eq("CNTNAP2_cohort4.csv") & data.session.lt(13)).any()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
assert selected.response_poke.isin([-1, 1]).all()
selected["right"] = selected.response_poke.eq(1)
selected["correct"] = selected.success.eq(1)
selected["group"] = selected.filename + "/" + selected.animal
observed = {"animals": selected.animal.nunique(), "trials": len(selected),
            "right": int(selected.right.sum()), "left": int((~selected.right).sum())}
assert observed == EXPECTED
counts = selected.groupby(["group", "filename", "animal", "genotype", "plot_ABL", "ILD"], as_index=False).agg(
    trials=("right", "size"), right_trials=("right", "sum"), correct_trials=("correct", "sum"))
counts["left_trials"] = counts.trials - counts.right_trials
counts["p_right"] = counts.right_trials / counts.trials
counts["fitted_p_right"] = np.nan
fit_rows = []
for (group, filename, animal, genotype, abl), condition in counts.groupby(["group", "filename", "animal", "genotype", "plot_ABL"], sort=False):
    condition = condition.sort_values("ILD")
    assert condition.ILD.tolist() == SIGNED_ILDS
    x = condition.ILD.to_numpy(dtype=float)
    result = fit_condition(x, condition.p_right.to_numpy())
    fit_rows.append({"view": "animal", "group": group, "filename": filename, "animal": animal,
        "genotype": genotype, "plot_ABL": int(abl), "animals": 1,
        "trials": int(condition.trials.sum()), "right_trials": int(condition.right_trials.sum()),
        "left_trials": int(condition.left_trials.sum()), "ILD_points": len(condition), **result})
    if result["fit_status"] == "ok":
        counts.loc[condition.index, "fitted_p_right"] = sigmoid(x, *[result[key] for key in ["upper", "lower", "x0", "k"]])
fits = pd.DataFrame(fit_rows)
assert len(fits) == 159 and fits.fit_status.eq("ok").all() and len(counts) == 1590
np.testing.assert_array_equal(fits.k, fits.slope_k)
counts["residual"] = counts.fitted_p_right - counts.p_right
counts.to_csv(OUTPUT / "slope_jnd_condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "slope_jnd_fits.csv", index=False)
selected.groupby(["filename", "animal", "genotype"], as_index=False).agg(
    trials=("right", "size"), right_trials=("right", "sum")).to_csv(OUTPUT / "animal_membership.csv", index=False)
print("Refitted 159 animal/ABL conditions:", json.dumps(observed), flush=True)


# %% Audit all animals; require a complete three-ABL set separately for each metric
eligibility = fits.groupby(["animal", "filename", "genotype"], as_index=False).agg(
    choice_trials=("trials", "sum"), right_trials=("right_trials", "sum"), left_trials=("left_trials", "sum"))
for metric, config in METRICS.items():
    wide = fits.pivot(index="animal", columns="plot_ABL", values=config["source"])
    wide.columns = [f"{metric}_ABL_{a}" for a in wide.columns]
    eligibility = eligibility.merge(wide.reset_index(), on="animal", validate="one_to_one")
    columns = [f"{metric}_ABL_{a}" for a in ABLS]
    eligibility[f"{metric}_eligible"] = np.isfinite(eligibility[columns]).all(axis=1) & eligibility[columns].gt(0).all(axis=1)
    eligibility[f"{metric}_unavailable_ABLs"] = eligibility.apply(
        lambda row: ";".join(str(a) for a in ABLS if not np.isfinite(row[f"{metric}_ABL_{a}"])), axis=1)
    assert eligibility[f"{metric}_eligible"].sum() == config["animals"]
assert set(eligibility.loc[~eligibility.jnd_eligible, "animal"]) == {"ASD0035", "ASD0044", "ASD0048", "ASD0058"}
eligibility.sort_values("animal").to_csv(OUTPUT / "animal_eligibility.csv", index=False)

means_tables, point_tables, summary_records = [], [], []
for metric, config in METRICS.items():
    selected_animals = eligibility.loc[eligibility[f"{metric}_eligible"], "animal"]
    values = fits.loc[fits.animal.isin(selected_animals)].copy()
    values["value"] = values[config["source"]]
    means = values.groupby(["animal", "filename", "genotype"], as_index=False).agg(
        animal_mean=("value", "mean"), choice_trials=("trials", "sum"),
        right_trials=("right_trials", "sum"), left_trials=("left_trials", "sum"))
    means = means.sort_values(["animal_mean", "animal"], kind="stable").reset_index(drop=True)
    means.insert(0, "metric", metric)
    means["position"] = np.arange(len(means))
    means["order_rank"] = means.position + 1
    means["units"] = config["units"]
    values = values.merge(means[["animal", "animal_mean", "position", "order_rank"]], on="animal", validate="many_to_one")
    values["signed_delta"] = values.value - values.animal_mean
    values["metric"] = metric
    values["units"] = config["units"]
    values = values[["metric", "animal", "filename", "genotype", "position", "order_rank", "plot_ABL",
        "value", "animal_mean", "signed_delta", "units", "trials", "right_trials", "left_trials"]].sort_values(["position", "plot_ABL"])
    assert len(values) == 3 * config["animals"] and means.choice_trials.sum() == config["trials"]
    np.testing.assert_allclose(values.groupby("animal").signed_delta.sum(), 0, rtol=0, atol=2e-14)
    for level, data in [("animal_means", means.animal_mean), ("ABL_deviations", values.signed_delta)]:
        center, sd = float(data.mean()), float(data.std(ddof=POPULATION_SD_DDOF))
        summary_records.append({"metric": metric, "level": level, "n": len(data), "mean": center,
            "population_sd": sd, "ddof": POPULATION_SD_DDOF, "lower": center - sd, "upper": center + sd,
            "units": config["units"]})
    means_tables.append(means)
    point_tables.append(values)
means_all = pd.concat(means_tables, ignore_index=True)
points = pd.concat(point_tables, ignore_index=True)
summaries = pd.DataFrame(summary_records)
assert len(means_all) == 102 and len(points) == 306 and len(summaries) == 4
means_all.to_csv(OUTPUT / "animal_means_order.csv", index=False)
points.to_csv(OUTPUT / "animal_abl_deviations.csv", index=False)
summaries.to_csv(OUTPUT / "summary_statistics.csv", index=False)


# %% Before/after comparisons: changes by animal/ABL, eligibility, and common sets
previous_fits = pd.read_csv(REFERENCE / "slope_jnd_fits.csv").query("view == 'animal'").copy()
previous_fits["animal"] = previous_fits.group.str.split("/").str[-1]
old_counts = pd.read_csv(REFERENCE / "slope_jnd_condition_counts.csv").query("view == 'animal'")
old_counts["animal"] = old_counts.group.str.split("/").str[-1]
condition_comparison = old_counts.merge(counts, on=["animal", "plot_ABL", "ILD"], suffixes=("_before", "_after"), validate="one_to_one")
for field in ["trials", "right_trials", "left_trials"]:
    condition_comparison[f"{field}_removed"] = condition_comparison[f"{field}_before"] - condition_comparison[f"{field}_after"]
    assert condition_comparison[f"{field}_removed"].ge(0).all()
condition_comparison.to_csv(OUTPUT / "condition_count_comparison.csv", index=False)
fields = ["animal", "plot_ABL", "trials", "right_trials", "left_trials", "upper", "lower", "x0", "k", "slope_k", "jnd", "jnd_status"]
comparison = previous_fits[fields].merge(fits[fields], on=["animal", "plot_ABL"], suffixes=("_before", "_after"), validate="one_to_one")
comparison["choices_removed"] = comparison.trials_before - comparison.trials_after
for field in ["upper", "lower", "x0", "k", "jnd"]:
    comparison[f"{field}_change"] = comparison[f"{field}_after"] - comparison[f"{field}_before"]
assert len(comparison) == 159 and comparison.choices_removed.sum() == 20035
changed_animals = sorted(comparison.loc[comparison.choices_removed.gt(0), "animal"].unique())
assert len(changed_animals) == 10
unchanged = comparison.choices_removed.eq(0)
assert unchanged.sum() == 129
np.testing.assert_allclose(comparison.loc[unchanged, ["upper_after", "lower_after", "x0_after", "k_after"]],
                           comparison.loc[unchanged, ["upper_before", "lower_before", "x0_before", "k_before"]], rtol=0, atol=1e-10)
comparison.to_csv(OUTPUT / "fit_comparison.csv", index=False)
old_means = pd.read_csv(PREVIOUS_PLOTS / "animal_means_order.csv")
old_values = pd.read_csv(PREVIOUS_PLOTS / "animal_abl_deviations.csv")
mean_comparison = old_means.merge(means_all, on=["metric", "animal", "filename", "genotype"], how="outer", suffixes=("_before", "_after"), validate="one_to_one")
mean_comparison["animal_mean_change"] = mean_comparison.animal_mean_after - mean_comparison.animal_mean_before
mean_comparison.to_csv(OUTPUT / "animal_mean_comparison.csv", index=False)
delta_comparison = old_values.merge(points, on=["metric", "animal", "plot_ABL"], how="outer", suffixes=("_before", "_after"), validate="one_to_one")
delta_comparison["signed_delta_change"] = delta_comparison.signed_delta_after - delta_comparison.signed_delta_before
delta_comparison.to_csv(OUTPUT / "signed_delta_comparison.csv", index=False)
summary_comparison = []
for metric in METRICS:
    before = old_values.loc[old_values.metric.eq(metric)]
    after = points.loc[points.metric.eq(metric)]
    common = set(before.animal) & set(after.animal)
    for population in ["all_metric_eligible", "same_animals_before_after"]:
        left = before if population == "all_metric_eligible" else before.loc[before.animal.isin(common)]
        right = after if population == "all_metric_eligible" else after.loc[after.animal.isin(common)]
        for level in ["animal_means", "ABL_deviations"]:
            v_before = left.groupby("animal").animal_mean.first() if level == "animal_means" else left.signed_delta
            v_after = right.groupby("animal").animal_mean.first() if level == "animal_means" else right.signed_delta
            m_before, m_after = float(v_before.mean()), float(v_after.mean())
            sd_before, sd_after = float(v_before.std(ddof=0)), float(v_after.std(ddof=0))
            summary_comparison.append({"metric": metric, "population": population, "level": level,
                "animals_before": left.animal.nunique(), "animals_after": right.animal.nunique(),
                "mean_before": m_before, "mean_after": m_after, "mean_change": m_after-m_before,
                "population_SD_before": sd_before, "population_SD_after": sd_after,
                "SD_change_percent": 100*(sd_after/sd_before-1), "units": METRICS[metric]["units"]})
summary_comparison = pd.DataFrame(summary_comparison)
summary_comparison.to_csv(OUTPUT / "summary_comparison.csv", index=False)
comparison_report = {"choices_before": 471416, "choices_after": 451381, "choices_removed": 20035,
    "changed_animals": changed_animals, "unchanged_animals": 43, "unchanged_animal_ABL_fits": 129,
    "jnd_eligible_before": 50, "jnd_eligible_after": 49, "newly_ineligible_jnd": ["ASD0058"],
    "slope_eligible_before": 53, "slope_eligible_after": 53,
    "population_comparisons": summary_comparison.to_dict("records")}
(OUTPUT / "comparison.json").write_text(json.dumps(comparison_report, indent=2) + "\n")
print(summary_comparison.to_string(index=False), flush=True)


# %% Two figures: values above, signed deviations below; SD bars at the right
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12})
figure_records, drawn_points, drawn_bars = [], [], []
for metric, config in METRICS.items():
    means = means_all.loc[means_all.metric.eq(metric)]
    values = points.loc[points.metric.eq(metric)]
    summary = summaries.loc[summaries.metric.eq(metric)].set_index("level")
    grand_mean = summary.loc["animal_means", "mean"]
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, .05], hspace=.29, wspace=.025,
                           left=.055, right=.975, top=.85, bottom=.255)
    upper = fig.add_subplot(grid[0, 0])
    lower = fig.add_subplot(grid[1, 0], sharex=upper)
    upper_sd = fig.add_subplot(grid[0, 1], sharey=upper)
    lower_sd = fig.add_subplot(grid[1, 1], sharey=lower)
    fig.suptitle(f"Session type 1 | Post-calibration exclusion | {config['title']} and signed deviation from each animal's mean", fontsize=18, y=.985)
    fig.text(.5, .947, f"{config['animals']} eligible animals | {config['trials']:,} valid choices | {len(values)} ABL values in each panel | sorted by animal mean",
             ha="center", fontsize=12)
    handles = [Line2D([], [], marker="o", color=COLORS[a], linestyle="None", label=f"ABL {a} dB") for a in ABLS]
    handles += [Line2D([], [], marker="_", color="black", linestyle="None", markersize=10, label="Animal mean (three ABLs)"),
                Line2D([], [], color="black", linestyle=":", label="Grand mean (equal animals)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925), ncol=5, frameon=False)
    for abl in ABLS:
        condition = values.loc[values.plot_ABL.eq(abl)]
        upper.scatter(condition.position, condition.value, color=COLORS[abl], s=27, marker="o", alpha=DOT_ALPHA, zorder=3)
        lower.scatter(condition.position, condition.signed_delta, color=COLORS[abl], s=27, marker="o", alpha=DOT_ALPHA, zorder=3)
    upper.scatter(means.position, means.animal_mean, marker="_", color="black", s=95, linewidths=1.5, zorder=5)
    upper.axhline(grand_mean, color="black", linestyle=":", linewidth=1)
    lower.axhline(0, color="black", linewidth=1)
    upper.set_title(f"{config['title']} at each ABL", loc="left")
    lower.set_title("Signed deviation: value at this ABL − animal mean", loc="left")
    upper.set_ylabel(f"{'JND' if metric == 'jnd' else 'k'} ({config['units']})")
    lower.set_ylabel(f"{'JND' if metric == 'jnd' else 'k'} − animal mean ({config['units']})")
    high = max(float(values.value.max()), float(summary.loc["animal_means", "upper"]))
    upper.set_ylim(0, np.ceil(high * 1.04 / config["raw_tick"]) * config["raw_tick"])
    extent = max(float(values.signed_delta.abs().max()), abs(float(summary.loc["ABL_deviations", "lower"])),
                 abs(float(summary.loc["ABL_deviations", "upper"])))
    extent = np.ceil(extent * 1.06 / config["delta_tick"]) * config["delta_tick"]
    lower.set_ylim(-extent, extent)
    labels = [f"{r.animal}\n{r.filename.removesuffix('.csv')}\n{GENOTYPE_LABELS[r.genotype]}" for r in means.itertuples()]
    lower.set_xticks(means.position, labels, rotation=90, fontsize=7)
    lower.set_xlabel("Animal | cohort | genotype (same order in both panels)")
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    for ax in [upper, lower]:
        ax.set_xlim(-.8, len(means) - .2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for panel, ax, level in [("raw", upper_sd, "animal_means"), ("delta", lower_sd, "ABL_deviations")]:
        row = summary.loc[level]
        ax.plot([.5, .5], [row.lower, row.upper], color="gray", linewidth=4, solid_capstyle="butt")
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.set_title(f"Mean ±1 SD\nσ = {row.population_sd:.4f}", fontsize=9)
        box = ax.get_position()
        ylim = ax.get_ylim()
        drawn_bars.append({"metric": metric, "panel": panel, "mean": row['mean'], "sd": row.population_sd,
            "lower": row.lower, "upper": row.upper, "pdf_x_pt": (box.x0 + .5 * box.width) * FIGSIZE[0] * 72,
            "pdf_y_top_pt": (1 - box.y0 - (row.upper - ylim[0])/(ylim[1] - ylim[0]) * box.height) * FIGSIZE[1] * 72,
            "pdf_y_bottom_pt": (1 - box.y0 - (row.lower - ylim[0])/(ylim[1] - ylim[0]) * box.height) * FIGSIZE[1] * 72})
    fig.text(.5, .075, "Animal mean = (M20 + M40 + M60) / 3 | Signed delta = M_ABL − animal mean | Gray side bars: mean ±1 population SD (ddof=0)",
             ha="center", fontsize=10)
    fig.text(.5, .044, "Valid choices: success in {1,−1}; training 16; no RT restriction; prepared repeat rule already applied; ABL 59 → 60; signed ILDs ±1/2/4/8/16",
             ha="center", fontsize=10)
    fig.text(.5, .016, "CNTNAP2 cohort 4: sessions < 13 removed for speaker calibration. Each animal/ABL has equal summary weight; signed deltas are not pairwise ranking scores.",
             ha="center", fontsize=10)
    for panel, ax, field in [("raw", upper, "value"), ("delta", lower, "signed_delta")]:
        b = ax.get_position()
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        for r in values.itertuples(index=False):
            value = getattr(r, field)
            drawn_points.append({"metric": metric, "panel": panel, "animal": r.animal, "plot_ABL": r.plot_ABL,
                "position": r.position, "value": value, "color": to_hex(COLORS[r.plot_ABL]),
                "pdf_x_pt": (b.x0 + (r.position-xlo)/(xhi-xlo)*b.width) * FIGSIZE[0] * 72,
                "pdf_y_pt": (1-b.y0-(value-ylo)/(yhi-ylo)*b.height) * FIGSIZE[1] * 72})
    name = f"ASD_session_type_1_calibrated_animal_{config['filename']}"
    pdf, png = OUTPUT / f"{name}.pdf", OUTPUT / f"{name}.png"
    fig.savefig(pdf, metadata={"Title": f"ASD: {config['title']} and signed animal-centered deviations"})
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    figure_records.append({"metric": metric, "animals": len(means), "choices": config['trials'],
        "ABL_values_per_panel": len(values), "animal_order": means.animal.to_list(),
        "raw_ylim": list(upper.get_ylim()), "delta_ylim": list(lower.get_ylim()),
        "pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO))})
    plt.close(fig)
    print(f"Saved {metric}: {len(means)} animals, {len(values)} values per panel", flush=True)
pd.DataFrame(drawn_points).to_csv(OUTPUT / "plotted_points.csv", index=False)
pd.DataFrame(drawn_bars).to_csv(OUTPUT / "plotted_summary_bars.csv", index=False)


# %% Validate preservation and record the new dataset/fit definitions
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "reference_folder": str(REFERENCE.relative_to(REPO)), "previous_plot_folder": str(PREVIOUS_PLOTS.relative_to(REPO)),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "selection": "Session type 1, training 16, success in {1,-1}; ABL59->60 then20/40/60; signed ILDs ±1/2/4/8/16; prepared repeat and short-sound rules; CNTNAP2 cohort4 session>=13",
    "refitted": True, "fit_count": 159, "condition_points": 1590,
    "fit_function": "lower + (upper-lower)/(1+exp(-k*(ILD-x0)))",
    "fit_parameters": ["upper", "lower", "x0", "k"], "fit_initial_guess": "[max(P), min(P), median(ILD), 0.1]",
    "fit_weighting": "Unweighted curve_fit; equal weight per ten measured ILD probabilities", "fit_maxfev": MAXFEV,
    "jnd_definition": "Half crossing distance at y_high=.75*upper and y_low=.25*(1-lower); retain undefined crossings as NaN",
    "rt_restriction": None, "repeat_filter_reapplied": False, "observed": observed,
    "animals_audited": 53, "eligible_animals": {"jnd": 49, "k": 53},
    "eligible_choices": {"jnd": 420616, "k": 451381}, "ABL_values_per_panel": {"jnd": 147, "k": 159},
    "eligibility": "Require finite positive values at all three ABLs independently for each metric; no incomplete means",
    "jnd_excluded_animals": ["ASD0035", "ASD0044", "ASD0048", "ASD0058"],
    "animal_mean": "Equal arithmetic mean of ABL20/40/60 values", "delta": "Signed M_ABL - animal mean",
    "grand_mean": "Equal arithmetic mean of eligible animal means", "summary_SD_ddof": POPULATION_SD_DDOF,
    "upper_summary": "Grand mean +/- population SD of animal means",
    "lower_summary": "Mean +/- population SD of all displayed signed ABL deviations",
    "sort": "Ascending unrounded animal mean; ties by animal ID", "colors": {str(a): to_hex(c) for a,c in COLORS.items()},
    "figure_inches": list(FIGSIZE), "png_dpi": PNG_DPI, "figures": figure_records,
    "audit_values": len(points), "plotted_values": len(drawn_points), "summary_bars": len(drawn_bars),
    "protected_file_sha256": protected_hashes}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved two updated PNG/PDF pairs, new fits/counts, signed deltas, eligibility and before/after audits", flush=True)
