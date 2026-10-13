# %% Imports
"""Eight scatter figures: fitted sigmoid k and Figure 1 JND for ASD session type 1.

Run .venv/bin/python asd_analysis/plot_session_type_1_slope_jnd.py.
Pool observed choice counts within each condition before fitting pooled views.
Fit all ten signed-ILD probabilities with equal weight, as in paper Figure 1.
"""
import hashlib
import json
from pathlib import Path
import warnings
from unittest.mock import patch

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_slope_jnd"
PSYCHOMETRICS = REPO / "asd_analysis/session_type_1_psychometrics"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "Heterozygous", "hom": "Homozygous"}
SHORT_GENOTYPES = {"wt": "WT", "het": "Het", "hom": "Hom"}
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
ANIMAL_DOT_OFFSETS = {20: -.22, 40: 0, 60: .22}
POOLED_CATEGORY_SPACING = 1.7
POOLED_LABEL_ROTATION = 45
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
PNG_DPI = 180
EXPECTED = {"animals": 53, "trials": 471416, "right": 241539, "left": 229877}
VIEWS = {
    "animal": {"column": "animal_group", "groups": None, "figsize": (26, 8), "fit_count": 159, "title": "Individual animals"},
    "filename_genotype": {"column": "filename", "groups": SOURCE_FILES, "figsize": (24, 8.5), "fit_count": 36, "title": "Filename + genotype"},
    "family_genotype": {"column": "family", "groups": FAMILIES, "figsize": (17, 8), "fit_count": 18, "title": "Gene family + genotype | families separate"},
    "genotype": {"column": None, "groups": ["all_cohorts"], "figsize": (12, 7), "fit_count": 9, "title": "Genotype only | both families combined"},
}
METRICS = {"slope_k": {"name": "slope", "label": "k (dB⁻¹)", "title": "Slope = k"},
           "jnd": {"name": "jnd", "label": "JND (dB)", "title": "Figure 1 JND"}}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Sigmoid, JND, and fitting: one fit supplies both metrics
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


# %% Meaningful mathematical and failure checks before fitting observed data
test_x = np.array(SIGNED_ILDS, dtype=float)
test_parameters = [.97, .03, .5, .4]
test_fit = fit_condition(test_x, sigmoid(test_x, *test_parameters))
assert test_fit["fit_status"] == "ok"
np.testing.assert_allclose([test_fit[key] for key in ["upper", "lower", "x0", "k"]], test_parameters, rtol=0, atol=1e-6)
assert test_fit["slope_k"] == test_fit["k"]
for test_k in [.1, .4, 2]:
    test_jnd = jnd_from_fit(1, 0, 2.3, test_k)
    assert np.isclose(test_jnd["jnd"], np.log(3) / test_k, rtol=0, atol=1e-12)
    for side in ["high", "low"]:
        assert np.isclose(sigmoid(test_jnd[f"ild_{side}"], 1, 0, 2.3, test_k), test_jnd[f"y_{side}"])
test_missing = fit_condition(test_x, sigmoid(test_x, .55, .35, .5, .4))
assert test_missing["fit_status"] == "ok" and np.isfinite(test_missing["slope_k"])
assert test_missing["jnd_status"] != "ok" and np.isnan(test_missing["jnd"])
for invalid in [(1, 0, 0, 0), (.3, .6, 0, .4), (.5, .5, 0, .4)]:
    assert jnd_from_fit(*invalid)["jnd_status"] == "invalid_fit"
    with patch(f"{__name__}.curve_fit", return_value=(np.array(invalid), np.eye(4), {"nfev": 1}, "synthetic fit", 1)):
        test_nonincreasing = fit_condition(test_x, sigmoid(test_x, *test_parameters))
    assert test_nonincreasing["fit_status"] == "non_increasing"
    assert np.isnan(test_nonincreasing["slope_k"]) and np.isnan(test_nonincreasing["jnd"])
with patch(f"{__name__}.curve_fit", side_effect=RuntimeError("synthetic optimizer failure")):
    test_failed = fit_condition(test_x, sigmoid(test_x, *test_parameters))
assert test_failed["fit_status"] == "optimizer_failed"
assert np.isnan(test_failed["slope_k"]) and np.isnan(test_failed["jnd"])


# %% Provenance: all existing CSVs, plots, and the October 12 book stay unchanged
old_folders = [REPO / "asd_analysis" / name for name in [
    "session_type_1_psychometrics", "session_type_1_rtds",
    "session_type_1_genotype_rtds", "session_type_1_ild_pooled_rtds",
]]
protected_paths = [REPO / name for name in SOURCE_FILES]
protected_paths += [INPUT.parent / f"ASD_session_type_{value}.csv" for value in [1, 2, 23]]
protected_paths += [path for folder in old_folders for path in folder.iterdir() if path.is_file()]
protected_paths += list((REPO / "docs/assets/results/2026-10-12").iterdir())
protected_paths += [REPO / "docs/results/2026-10-12.md"]
protected_hashes = {}
for path in protected_paths:
    if path.is_file():
        with path.open("rb") as stream:
            protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
reference = json.loads((PSYCHOMETRICS / "psychometric_validation.json").read_text())
assert reference["status"] == "passed" and input_sha256 == reference["input_sha256"]


# %% Same choices as the psychometrics, retaining missing RT trials
data = pd.read_csv(INPUT, usecols=[
    "filename", "animal", "genotype", "session_type", "training_level", "trial_is_repeat",
    "is_short_sound", "short_duration", "ABL", "ILD", "success", "response_poke", "timed_rt",
], low_memory=False)
assert len(data) == 642370 and data["animal"].nunique() == 53
assert data["session_type"].eq(1).all() and data["training_level"].eq(16).all()
assert data["trial_is_repeat"].eq(False).all()
assert data["is_short_sound"].eq(0).all() and data["short_duration"].eq(0).all()
data["plot_ABL"] = data["ABL"].replace({59: 60})
selected = data.loc[data["success"].isin([1, -1]) & data["plot_ABL"].isin(ABLS)
                    & data["ILD"].isin(SIGNED_ILDS)].copy()
assert selected["response_poke"].isin([-1, 1]).all()
selected["right"] = selected["response_poke"].eq(1)
selected["correct"] = selected["success"].eq(1)
selected["family"] = selected["filename"].str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
selected["animal_group"] = selected["filename"] + "/" + selected["animal"]
assert selected["genotype"].isin(GENOTYPES).all() and selected["family"].isin(FAMILIES).all()
assert selected.groupby("animal")["filename"].nunique().eq(1).all()
assert selected.groupby("animal")["genotype"].nunique().eq(1).all()
assert {"ASD0026", "ASD0045"} <= set(selected["animal"])
observed = {"animals": selected["animal"].nunique(), "trials": len(selected),
            "right": int(selected["right"].sum()), "left": int((~selected["right"]).sum())}
assert observed == EXPECTED
animal_order = selected[["filename", "animal", "genotype", "animal_group"]].drop_duplicates().copy()
animal_order["cohort_order"] = animal_order["filename"].map({name: i for i, name in enumerate(SOURCE_FILES)})
animal_order["genotype_order"] = animal_order["genotype"].map({name: i for i, name in enumerate(GENOTYPES)})
animal_order = animal_order.sort_values(["cohort_order", "genotype_order", "animal"]).reset_index(drop=True)
VIEWS["animal"]["groups"] = animal_order["animal_group"].tolist()
print("Confirmed choices:", json.dumps(observed), flush=True)


# %% Aggregate counts before fitting; keep condition, group, and membership audits
point_parts, group_parts, member_parts = [], [], []
for view, config in VIEWS.items():
    rows = selected.copy()
    rows["group"] = rows[config["column"]] if config["column"] else "all_cohorts"
    points = rows.groupby(["group", "genotype", "plot_ABL", "ILD"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"),
        correct_trials=("correct", "sum"), animals=("animal", "nunique"),
    )
    points["left_trials"] = points["trials"] - points["right_trials"]
    points["error_trials"] = points["trials"] - points["correct_trials"]
    points["p_right"] = points["right_trials"] / points["trials"]
    points.insert(0, "view", view)
    assert len(points) == config["fit_count"] * 10
    assert points[["trials", "right_trials", "left_trials"]].sum().tolist() == [471416, 241539, 229877]
    point_parts.append(points)
    counts = rows.groupby(["group", "genotype"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"), animals=("animal", "nunique"))
    counts.insert(0, "view", view)
    group_parts.append(counts)
    members = rows.groupby(["group", "genotype", "filename", "animal"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"))
    members.insert(0, "view", view)
    assert len(members) == 53
    member_parts.append(members)
points = pd.concat(point_parts, ignore_index=True)
groups = pd.concat(group_parts, ignore_index=True)
membership = pd.concat(member_parts, ignore_index=True)
count_columns = ["trials", "right_trials", "left_trials", "correct_trials", "error_trials", "animals"]
keys = ["group", "genotype", "plot_ABL", "ILD"]
old_points = pd.read_csv(PSYCHOMETRICS / "psychometric_points.csv")
for view in ["filename_genotype", "genotype"]:
    pd.testing.assert_frame_equal(points.loc[points["view"].eq(view)].set_index(keys)[count_columns].sort_index(),
                                   old_points.loc[old_points["view"].eq(view)].set_index(keys)[count_columns].sort_index())
family_points = pd.read_csv(PSYCHOMETRICS / "family_psychometric_points.csv").rename(columns={"family": "group"})
pd.testing.assert_frame_equal(points.loc[points["view"].eq("family_genotype")].set_index(keys)[count_columns].sort_index(),
                               family_points.set_index(keys)[count_columns].sort_index())
animal_points = points.loc[points["view"].eq("animal")].copy()
animal_points["group"] = animal_points["group"].str.split("/").str[0]
pd.testing.assert_frame_equal(animal_points.groupby(keys)[count_columns].sum().sort_index(),
                               points.loc[points["view"].eq("filename_genotype")].set_index(keys)[count_columns].sort_index())
groups.to_csv(OUTPUT / "slope_jnd_group_counts.csv", index=False)
membership.to_csv(OUTPUT / "slope_jnd_group_membership.csv", index=False)


# %% Fit the ten equally weighted signed-ILD points for every group and ABL
fit_rows = []
points["fitted_p_right"] = np.nan
for (view, group, genotype, abl), condition in points.groupby(["view", "group", "genotype", "plot_ABL"], sort=False):
    condition = condition.sort_values("ILD")
    assert condition["ILD"].tolist() == SIGNED_ILDS
    x = condition["ILD"].to_numpy(dtype=float)
    probability = condition["p_right"].to_numpy()
    result = fit_condition(x, probability)
    animal_count = int(groups.loc[groups["view"].eq(view) & groups["group"].eq(group)
                                 & groups["genotype"].eq(genotype), "animals"].iloc[0])
    fit_rows.append({"view": view, "group": group, "genotype": genotype, "plot_ABL": int(abl),
                     "animals": animal_count, "trials": int(condition["trials"].sum()),
                     "right_trials": int(condition["right_trials"].sum()),
                     "left_trials": int(condition["left_trials"].sum()), "ILD_points": len(condition), **result})
    if np.isfinite([result[key] for key in ["upper", "lower", "x0", "k"]]).all():
        points.loc[condition.index, "fitted_p_right"] = sigmoid(x, *[result[key] for key in ["upper", "lower", "x0", "k"]])
fits = pd.DataFrame(fit_rows)
assert len(fits) == 222 and len(groups) == 74 and len(points) == 2220
assert fits.groupby("view").size().to_dict() == {view: config["fit_count"] for view, config in VIEWS.items()}
assert fits.loc[fits["fit_status"].eq("ok"), "slope_k"].equals(fits.loc[fits["fit_status"].eq("ok"), "k"])
points["residual"] = points["fitted_p_right"] - points["p_right"]
points.to_csv(OUTPUT / "slope_jnd_condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "slope_jnd_fits.csv", index=False)
status = fits.groupby(["view", "fit_status", "jnd_status"]).size().rename("count").reset_index()
status.to_csv(OUTPUT / "slope_jnd_fit_status_counts.csv", index=False)
print("Fit status:\n" + status.to_string(index=False), flush=True)


# %% Scatter figures: one categorical axis for each of the four views
pages = []
legend = [Line2D([], [], color=ABL_COLORS[abl], marker=ABL_MARKERS[abl], linestyle="None",
                 label=f"ABL {abl} dB", markersize=7) for abl in ABLS]
for metric, metric_config in METRICS.items():
    for view, config in VIEWS.items():
        view_fits = fits.loc[fits["view"].eq(view)]
        valid = np.isfinite(view_fits[metric])
        missing = int((~valid).sum())
        maximum = float(view_fits.loc[valid, metric].max()) if valid.any() else 1.0
        ylim = (0, maximum * 1.22)
        categories = []
        if view == "animal":
            fig, ax = plt.subplots(figsize=config["figsize"])
            fig.suptitle(f"Session type 1 | {metric_config['title']} | 53 individual animals", fontsize=18, y=.985)
            fig.text(.5, .93, f"471,416 choices; {int(valid.sum())}/159 available estimates; {missing} unavailable", ha="center", fontsize=11)
            positions = {group: i for i, group in enumerate(config["groups"])}
            for abl in ABLS:
                observations = view_fits.loc[view_fits["plot_ABL"].eq(abl) & np.isfinite(view_fits[metric])]
                x = observations["group"].map(positions).to_numpy() + ANIMAL_DOT_OFFSETS[abl]
                scatter = ax.scatter(x, observations[metric], color=ABL_COLORS[abl], marker=ABL_MARKERS[abl], s=38, zorder=3)
                np.testing.assert_allclose(scatter.get_offsets(), np.column_stack([x, observations[metric]]))
                assert to_hex(scatter.get_facecolors()[0]) == to_hex(ABL_COLORS[abl])
            labels = []
            for row in animal_order.itertuples():
                info = groups.loc[groups["view"].eq("animal") & groups["group"].eq(row.animal_group)].iloc[0]
                labels.append(f"{row.animal}\n{SHORT_GENOTYPES[row.genotype]}\nn={int(info['trials']):,}")
            ax.set_xticks(np.arange(53), labels, rotation=90, fontsize=8)
            ax.set_xlim(-.7, 52.7)
            for filename in SOURCE_FILES:
                cohort = animal_order.index[animal_order["filename"].eq(filename)].to_numpy()
                start, stop = int(cohort[0]), int(cohort[-1])
                ax.text((start + stop) / 2, 1.025, f"{filename.removesuffix('.csv')} ({len(cohort)} animals)",
                        transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
                if start:
                    ax.axvline(start - .5, color=".7", linestyle=":", linewidth=.8)
            ax.set_ylim(*ylim)
            ax.set_ylabel(metric_config["label"], fontsize=13)
            ax.set_xlabel("Animal, genotype, and total eligible choice count", fontsize=11)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            assert len(ax.collections) == 3 and len(ax.lines) == 3
            assert all(np.ptp(line.get_xdata()) == 0 for line in ax.lines)
            fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .90), ncol=3, frameon=False)
            fig.subplots_adjust(left=.045, right=.99, top=.80, bottom=.29)
            panels = 1
        else:
            fig, ax = plt.subplots(figsize=config["figsize"])
            fig.suptitle(f"Session type 1 | {metric_config['title']} | {config['title']}", fontsize=16, y=.985)
            fig.text(.5, .925, f"53 animals | 471,416 choices | {int(valid.sum())}/{len(view_fits)} available estimates | {missing} unavailable",
                     ha="center", fontsize=11)
            fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .89), ncol=3, frameon=False)
            positions = {}
            labels = []
            for group in config["groups"]:
                for genotype in GENOTYPES:
                    info = groups.loc[groups["view"].eq(view) & groups["group"].eq(group) & groups["genotype"].eq(genotype)].iloc[0]
                    group_fits = view_fits.loc[view_fits["group"].eq(group) & view_fits["genotype"].eq(genotype)]
                    group_missing = int((~np.isfinite(group_fits[metric])).sum())
                    x = len(categories) * POOLED_CATEGORY_SPACING
                    positions[(group, genotype)] = x
                    category = f"{group.removesuffix('.csv')} | {GENOTYPE_LABELS[genotype]}" if view != "genotype" else GENOTYPE_LABELS[genotype]
                    animal_word = "animal" if int(info["animals"]) == 1 else "animals"
                    label = f"{category}\n{int(info['animals'])} {animal_word} | n={int(info['trials']):,}"
                    if group_missing:
                        label += f"\nunavailable={group_missing}/3"
                    labels.append(label)
                    categories.append({"group": group, "genotype": genotype, "x": x, "label": label,
                                       "animals": int(info["animals"]), "trials": int(info["trials"]),
                                       "unavailable_estimates": group_missing})
            for abl in ABLS:
                observations = view_fits.loc[view_fits["plot_ABL"].eq(abl) & np.isfinite(view_fits[metric])]
                x = np.array([positions[(row.group, row.genotype)] for row in observations.itertuples()]) + ANIMAL_DOT_OFFSETS[abl]
                scatter = ax.scatter(x, observations[metric], color=ABL_COLORS[abl], marker=ABL_MARKERS[abl], s=55, zorder=3)
                np.testing.assert_allclose(scatter.get_offsets(), np.column_stack([x, observations[metric]]))
                assert to_hex(scatter.get_facecolors()[0]) == to_hex(ABL_COLORS[abl])
            tick_positions = [category["x"] for category in categories]
            ax.set_xticks(tick_positions, labels, rotation=POOLED_LABEL_ROTATION,
                          ha="right", rotation_mode="anchor", fontsize=11)
            ax.tick_params(axis="x", pad=8)
            ax.set_xlim(-.85 * POOLED_CATEGORY_SPACING, tick_positions[-1] + .6 * POOLED_CATEGORY_SPACING)
            ax.set_ylim(*ylim)
            ax.set_ylabel(metric_config["label"], fontsize=13)
            axis_label = {"filename_genotype": "Filename + genotype", "family_genotype": "Gene family + genotype",
                          "genotype": "Genotype (both gene families and all cohorts combined)"}[view]
            ax.set_xlabel(axis_label, fontsize=12, labelpad=12)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            fig.subplots_adjust(left=.065, right=.99, top=.79, bottom=.42)
            assert len(fig.axes) == 1 and len(ax.collections) == 3 and len(ax.lines) == 0
            assert all(label.get_rotation() == POOLED_LABEL_ROTATION for label in ax.get_xticklabels())
            panels = 1
        fig.text(.5, .02,
                 "Session type 1; training 16; confirmed repeat rule; 807 short-sound rows removed during preparation; all 53 animals retained.\n"
                 "Correct/error choices; no RT restriction. ABL 59 → 60; signed ILDs ±1/2/4/8/16. Pool counts before fitting; each ILD probability point has equal fit weight.\n"
                 "Slope = k. JND = half the ILD spacing at y_high = 0.75 × upper and y_low = 0.25 × (1 − lower). Same fit for both metrics; no parameter averages.",
                 ha="center", va="bottom", fontsize=9)
        prefix = f"ASD_session_type_1_{view}_{metric_config['name']}"
        pdf = OUTPUT / f"{prefix}.pdf"
        png = OUTPUT / f"{prefix}.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=PNG_DPI)
        plt.close(fig)
        with fitz.open(pdf) as document:
            assert document.page_count == 1
            assert metric_config["title"] in document[0].get_text()
        pages.append({"view": view, "metric": metric, "panels": panels, "available_estimates": int(valid.sum()),
                      "unavailable_estimates": missing, "ylim": ylim, "pdf_pages": 1,
                      "categorical_groups": categories, "label_rotation": 90 if view == "animal" else POOLED_LABEL_ROTATION,
                      "category_spacing": 1 if view == "animal" else POOLED_CATEGORY_SPACING,
                      "pdf": str(pdf.relative_to(REPO)), "png": str(png.relative_to(REPO))})


# %% Final numeric validation and preservation manifest
for name, digest in protected_hashes.items():
    with (REPO / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
assert len(pages) == 8
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256, "prepared_rows": len(data),
    "selection": reference["selection"], "short_sound_cleanup": reference["short_sound_cleanup"],
    "observed_counts_per_view": observed, "included_nonfinite_timed_rt_trials": int((~np.isfinite(selected["timed_rt"])).sum()),
    "ABLs": ABLS, "ABL_mapping": {"59": 60}, "signed_ILDS": SIGNED_ILDS,
    "genotype_order": GENOTYPES, "source_file_order": SOURCE_FILES,
    "pooling": "Raw right/left counts pooled by signed ILD and ABL before fitting; no animal parameter/probability averages",
    "fit_weighting": "Unweighted least squares: equal weight for each of the ten signed-ILD P(right) points; sigma=None",
    "sigmoid": "lower + (upper-lower)/(1+exp(-k*(ILD-x0)))", "parameter_order": ["upper", "lower", "x0", "k"],
    "fit_bounds": {"upper": [0, 1], "lower": [0, 1], "x0": "unrestricted", "k": [0, "infinity"]},
    "initial_guess": "[max(Pright), min(Pright), median(signed ILDs), 0.1]", "maxfev": MAXFEV,
    "slope_definition": "fitted k, not midpoint derivative", "jnd_definition": "(inverse(0.75*upper)-inverse(0.25*(1-lower)))/2",
    "jnd_threshold_policy": "Require lower < y_low < y_high < upper and finite positive crossing spacing; retain valid k if JND is unavailable",
    "fits_attempted": len(fits), "condition_points": len(points),
    "groups_per_view": groups.groupby("view").size().to_dict(),
    "fits_per_view": fits.groupby("view").size().to_dict(), "fit_status_counts": status.to_dict(orient="records"),
    "ABL_colors": {str(abl): to_hex(color) for abl, color in ABL_COLORS.items()}, "ABL_markers": ABL_MARKERS,
    "animal_group_order": VIEWS["animal"]["groups"], "animal_dot_offsets": ANIMAL_DOT_OFFSETS,
    "scatter_only": True, "no_error_bars_or_mean_markers": True, "pages": pages,
    "synthetic_checks": ["recover known four parameters", "slope exactly k", "zero-lapse JND=ln(3)/k",
                         "sigmoid crossings match thresholds", "valid k retained when JND undefined",
                         "zero-k/non-increasing fits rejected", "optimizer failure leaves both metrics unavailable"],
    "all_pooled_conditions_match_existing_psychometric_audits": True,
    "all_animal_condition_counts_sum_to_filename_genotype_counts": True,
    "protected_file_sha256_unchanged": protected_hashes,
}
(OUTPUT / "slope_jnd_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved eight PNGs and eight single-page PDFs; all existing CSVs and figures are unchanged.", flush=True)
