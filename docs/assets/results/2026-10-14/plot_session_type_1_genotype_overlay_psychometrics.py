# %% Imports
"""Compare WT/HET/HOM after pooling choices or equally averaging ABL probabilities.

Run .venv/bin/python asd_analysis/plot_session_type_1_genotype_overlay_psychometrics.py.
Average measured probabilities first; fit the resulting ten points afterward.
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


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_overlay_psychometrics"
REFERENCE = REPO / "asd_analysis/session_type_1_psychometrics"
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
COLORS = {"wt": "tab:blue", "het": "tab:orange", "hom": "tab:green"}
MARKERS = {"wt": "o", "het": "s", "hom": "^"}
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
METHODS = {
    "trial_pooled": {"title": "Method A: pool trials across ABLs",
                     "definition": "sum(right choices) / sum(all choices) across ABL 20/40/60 at each ILD"},
    "equal_abl_average": {"title": "Method B: equally average ABL probabilities",
                          "definition": "arithmetic mean of measured P(right) at ABL 20/40/60, separately at each ILD; average first, then fit"},
}
VIEWS = {
    "filename_genotype": {"column": "filename", "groups": SOURCE_FILES, "shape": (2, 2),
                          "figsize": (14, 10), "title": "Four cohorts separately", "curves": 12},
    "family_genotype": {"column": "family", "groups": FAMILIES, "shape": (1, 2),
                        "figsize": (16, 6), "title": "SHANK3 and CNTNAP2 separately", "curves": 6},
    "genotype": {"column": None, "groups": ["all_cohorts"], "shape": (1, 1),
                 "figsize": (12, 7), "title": "Both families and all cohorts combined", "curves": 3},
}
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
XLIM = (-18, 18)
YLIM = (0, 1)
CURVE_ILDS = np.linspace(*XLIM, 501)
PNG_DPI = 180
EXPECTED = {"animals": 53, "trials": 471416, "right": 241539, "left": 229877}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Sigmoid fit: match the existing paper procedure without importing plot scripts
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper-lower) / (1 + np.exp(-k*(x-x0)))


def fit_sigmoid(x, probability):
    initial = [float(np.max(probability)), float(np.min(probability)), float(np.median(x)), INITIAL_K]
    result = {"upper": np.nan, "lower": np.nan, "x0": np.nan, "k": np.nan,
              "fit_status": "optimizer_failed", "initial_parameters": json.dumps(initial),
              "optimizer_code": 0, "optimizer_message": "", "function_evaluations": 0,
              "warnings": "", "covariance_finite": False, "unweighted_sse": np.nan, "rmse": np.nan}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, code = curve_fit(
                sigmoid, x, probability, p0=initial, bounds=FIT_BOUNDS,
                maxfev=MAXFEV, full_output=True)
        result.update(zip(["upper", "lower", "x0", "k"], parameters))
        result.update(optimizer_code=int(code), optimizer_message=message,
                      function_evaluations=int(info["nfev"]), covariance_finite=bool(np.isfinite(covariance).all()),
                      warnings="; ".join(dict.fromkeys(str(w.message) for w in caught)))
        if not np.isfinite(parameters).all():
            result["fit_status"] = "nonfinite_parameters"
            return result
        residual = sigmoid(x, *parameters)-probability
        result.update(unweighted_sse=float(residual @ residual), rmse=float(np.sqrt(np.mean(residual**2))))
        result["fit_status"] = "ok" if parameters[0] > parameters[1] and parameters[3] > 0 else "non_increasing"
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["optimizer_message"] = str(error)
    return result


# %% Meaningful checks for different ABL weighting and unavailable fits
toy_n = np.array([100, 10, 10])
toy_right = np.array([10, 8, 9])
assert np.isclose(toy_right.sum()/toy_n.sum(), .225)
assert np.isclose(np.mean(toy_right/toy_n), .6)
test_x = np.array(ILDS, dtype=float)
known = [.97, .03, .5, .4]
test_fit = fit_sigmoid(test_x, sigmoid(test_x, *known))
assert test_fit["fit_status"] == "ok"
np.testing.assert_allclose([test_fit[key] for key in ["upper", "lower", "x0", "k"]], known, atol=1e-6, rtol=0)
with patch(f"{__name__}.curve_fit", side_effect=RuntimeError("synthetic optimizer failure")):
    failed = fit_sigmoid(test_x, sigmoid(test_x, *known))
assert failed["fit_status"] == "optimizer_failed" and np.isnan(failed["k"])
for invalid in [[.3, .6, 0, .4], [1, 0, 0, 0]]:
    with patch(f"{__name__}.curve_fit", return_value=(np.array(invalid), np.eye(4), {"nfev": 1}, "synthetic fit", 1)):
        non_increasing = fit_sigmoid(test_x, sigmoid(test_x, *known))
    assert non_increasing["fit_status"] == "non_increasing"


# %% Snapshot inputs and previous analyses; every protected file must stay unchanged
protected = [REPO/name for name in SOURCE_FILES] + [REPO/"metadata.csv"]
protected += list(INPUT.parent.rglob("*.csv")) + list(INPUT.parent.rglob("*.json"))
for folder in ["session_type_1_psychometrics", "session_type_1_rtds", "session_type_1_genotype_rtds",
               "session_type_1_ild_pooled_rtds", "session_type_1_slope_jnd"]:
    protected += [p for p in (REPO/"asd_analysis"/folder).iterdir() if p.is_file()]
for date in ["2026-10-12", "2026-10-13"]:
    protected += list((REPO/"docs/assets/results"/date).iterdir())
    protected += [REPO/"docs/results"/f"{date}.md"]
protected += list((REPO/"asd_analysis").glob("*session_type*.py"))
protected_hashes = {}
for path in set(protected):
    if path.is_file() and path != Path(__file__):
        with path.open("rb") as stream:
            protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_hash = protected_hashes[str(INPUT.relative_to(REPO))]
reference = json.loads((REFERENCE/"psychometric_validation.json").read_text())
assert reference["status"] == "passed" and input_hash == reference["input_sha256"]


# %% Read the cleaned CSV; choices without an RT restriction
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "short_duration", "ABL", "ILD", "success", "response_poke", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all()
assert data.is_short_sound.eq(0).all() and data.short_duration.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(ILDS)].copy()
assert selected.response_poke.isin([1, -1]).all()
assert selected.genotype.isin(GENOTYPES).all() and set(selected.filename) == set(SOURCE_FILES)
selected["right"] = selected.response_poke.eq(1)
selected["correct"] = selected.success.eq(1)
selected["family"] = selected.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
observed = {"animals": selected.animal.nunique(), "trials": len(selected),
            "right": int(selected.right.sum()), "left": int((~selected.right).sum())}
assert observed == EXPECTED and {"ASD0026", "ASD0045"} <= set(selected.animal)
assert selected.response_poke.eq(np.sign(selected.ILD)*selected.success).all()
print("Confirmed choices:", json.dumps(observed), flush=True)


# %% Aggregate observed counts within each grouping, genotype, ABL, and signed ILD
abl_parts, condition_parts, group_parts, member_parts = [], [], [], []
for view, config in VIEWS.items():
    rows = selected.copy()
    rows["group"] = rows[config["column"]] if config["column"] else "all_cohorts"
    for columns, destination in [(["group", "genotype", "plot_ABL", "ILD"], abl_parts),
                                 (["group", "genotype", "ILD"], condition_parts),
                                 (["group", "genotype"], group_parts)]:
        counts = rows.groupby(columns, as_index=False).agg(trials=("response_poke", "size"),
            right_trials=("right", "sum"), correct_trials=("correct", "sum"), animals=("animal", "nunique"))
        counts["left_trials"] = counts.trials-counts.right_trials
        counts["error_trials"] = counts.trials-counts.correct_trials
        counts.insert(0, "view", view)
        destination.append(counts)
    members = rows.groupby(["group", "genotype", "filename", "animal"], as_index=False).agg(
        trials=("response_poke", "size"), right_trials=("right", "sum"))
    members.insert(0, "view", view)
    assert len(members) == 53
    member_parts.append(members)
abl_counts = pd.concat(abl_parts, ignore_index=True)
abl_counts["p_right"] = abl_counts.right_trials/abl_counts.trials
condition_counts = pd.concat(condition_parts, ignore_index=True)
groups = pd.concat(group_parts, ignore_index=True)
membership = pd.concat(member_parts, ignore_index=True)
assert len(abl_counts) == 630 and len(condition_counts) == 210 and len(groups) == 21 and len(membership) == 159
keys = ["group", "genotype", "plot_ABL", "ILD"]
count_fields = ["trials", "right_trials", "left_trials", "correct_trials", "error_trials", "animals"]
old_points = pd.read_csv(REFERENCE/"psychometric_points.csv")
for view in ["filename_genotype", "genotype"]:
    pd.testing.assert_frame_equal(abl_counts.loc[abl_counts.view.eq(view)].set_index(keys)[count_fields].sort_index(),
                                  old_points.loc[old_points.view.eq(view)].set_index(keys)[count_fields].sort_index())
old_family = pd.read_csv(REFERENCE/"family_psychometric_points.csv").rename(columns={"family": "group"})
pd.testing.assert_frame_equal(abl_counts.loc[abl_counts.view.eq("family_genotype")].set_index(keys)[count_fields].sort_index(),
                              old_family.set_index(keys)[count_fields].sort_index())
old_members = pd.read_csv(REFERENCE/"psychometric_group_membership.csv")
old_family_members = pd.read_csv(REFERENCE/"family_psychometric_group_membership.csv").rename(columns={"family": "group"})
old_family_members.insert(0, "view", "family_genotype")
member_keys = ["view", "group", "genotype", "filename", "animal"]
pd.testing.assert_frame_equal(membership.set_index(member_keys).sort_index(),
                              pd.concat([old_members, old_family_members]).set_index(member_keys).sort_index())
for view in VIEWS:
    assert groups.loc[groups.view.eq(view), ["trials", "right_trials", "left_trials", "animals"]].sum().tolist() == [471416, 241539, 229877, 53]
abl_counts.to_csv(OUTPUT/"overlay_abl_counts.csv", index=False)
groups.to_csv(OUTPUT/"overlay_group_counts.csv", index=False)
membership.to_csv(OUTPUT/"overlay_group_membership.csv", index=False)


# %% Collapse ABLs two ways: raw count fractions versus equal means of probabilities
point_rows = []
cell_keys = ["view", "group", "genotype", "ILD"]
raw_cells = condition_counts.set_index(cell_keys)
for cell_key, cell in abl_counts.groupby(cell_keys, sort=False):
    cell = cell.sort_values("plot_ABL")
    assert cell.plot_ABL.tolist() == ABLS and cell.trials.gt(0).all()
    raw = raw_cells.loc[cell_key]
    assert cell[["trials", "right_trials", "left_trials"]].sum().tolist() == raw[["trials", "right_trials", "left_trials"]].tolist()
    for method in METHODS:
        if method == "trial_pooled":
            probability = float(raw.right_trials/raw.trials)
            weights = cell.trials.to_numpy()/raw.trials
        else:
            probability = float(cell.p_right.mean())
            weights = np.full(3, 1/3)
        assert np.isclose(probability, weights @ cell.p_right.to_numpy(), rtol=0, atol=2e-15)
        record = dict(zip(cell_keys, cell_key))
        record.update(method=method, **{name: int(raw[name]) for name in count_fields}, p_right=probability)
        for index, abl in enumerate(ABLS):
            source = cell.iloc[index]
            record.update({f"ABL_{abl}_trials": int(source.trials), f"ABL_{abl}_right_trials": int(source.right_trials),
                           f"ABL_{abl}_p_right": float(source.p_right), f"ABL_{abl}_weight": float(weights[index])})
        point_rows.append(record)
points = pd.DataFrame(point_rows)
assert len(points) == 420 and points.p_right.between(0, 1).all()
points["fitted_p_right"] = np.nan


# %% Fit the ten combined ILD points, equally weighted in both methods
fit_rows = []
for fit_keys, condition in points.groupby(["method", "view", "group", "genotype"], sort=False):
    condition = condition.sort_values("ILD")
    assert condition.ILD.tolist() == ILDS
    result = fit_sigmoid(condition.ILD.to_numpy(dtype=float), condition.p_right.to_numpy())
    method, view, group, genotype = fit_keys
    info = groups.loc[groups.view.eq(view) & groups.group.eq(group) & groups.genotype.eq(genotype)].iloc[0]
    fit_rows.append({"method": method, "view": view, "group": group, "genotype": genotype,
                    "animals": int(info.animals), "trials": int(info.trials), "right_trials": int(info.right_trials),
                    "left_trials": int(info.left_trials), "ILD_points": 10, **result})
    if result["fit_status"] == "ok":
        points.loc[condition.index, "fitted_p_right"] = sigmoid(condition.ILD.to_numpy(),
            *[result[key] for key in ["upper", "lower", "x0", "k"]])
fits = pd.DataFrame(fit_rows)
points["residual"] = points.fitted_p_right-points.p_right
assert len(fits) == 42 and fits.groupby("method").size().eq(21).all()
points.to_csv(OUTPUT/"overlay_combined_points.csv", index=False)
fits.to_csv(OUTPUT/"overlay_fits.csv", index=False)
status = fits.groupby(["method", "view", "fit_status"]).size().rename("count").reset_index()
status.to_csv(OUTPUT/"overlay_fit_status_counts.csv", index=False)
print("Fit status:\n"+status.to_string(index=False), flush=True)


# %% Six figures: genotype colors, empirical points, and solid fitted sigmoids
pages = []
legend = [Line2D([], [], color=COLORS[g], marker=MARKERS[g], linestyle="-", label=LABELS[g], markersize=6) for g in GENOTYPES]
for method, method_config in METHODS.items():
    for view, config in VIEWS.items():
        fig, axes = plt.subplots(*config["shape"], figsize=config["figsize"], sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f"Session type 1 | Genotype psychometrics\n{config['title']}\n{method_config['title']}", fontsize=15, y=.985)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .905 if view == "filename_genotype" else .87), ncol=3, frameon=False)
        panel_audits = []
        for group, ax in zip(config["groups"], axes.flat):
            panel_groups = groups.loc[groups.view.eq(view) & groups.group.eq(group)]
            group_label = group.removesuffix(".csv") if group != "all_cohorts" else "All four cohorts"
            ax.set_title(f"{group_label}\n{int(panel_groups.animals.sum())} animals | {int(panel_groups.trials.sum()):,} eligible choices", fontsize=11, pad=10)
            panel_legend = []
            valid_fits = 0
            for genotype in GENOTYPES:
                condition = points.loc[points.method.eq(method) & points.view.eq(view) & points.group.eq(group)
                                       & points.genotype.eq(genotype)].sort_values("ILD")
                estimate = fits.loc[fits.method.eq(method) & fits.view.eq(view) & fits.group.eq(group)
                                    & fits.genotype.eq(genotype)].iloc[0]
                scatter = ax.scatter(condition.ILD, condition.p_right, color=COLORS[genotype], marker=MARKERS[genotype],
                                     s=33, alpha=.9, zorder=3)
                np.testing.assert_allclose(scatter.get_offsets(), condition[["ILD", "p_right"]].to_numpy())
                assert to_hex(scatter.get_facecolors()[0]) == to_hex(COLORS[genotype])
                available = estimate.fit_status == "ok"
                if available:
                    parameters = [estimate[key] for key in ["upper", "lower", "x0", "k"]]
                    curve = sigmoid(CURVE_ILDS, *parameters)
                    assert np.isfinite(curve).all() and ((curve >= 0) & (curve <= 1)).all()
                    ax.plot(CURVE_ILDS, curve, color=COLORS[genotype], linestyle="-", linewidth=1.8, zorder=2)
                    valid_fits += 1
                animal_word = "animal" if estimate.animals == 1 else "animals"
                label = f"{LABELS[genotype]}: {int(estimate.animals)} {animal_word}, n={int(estimate.trials):,}"
                if not available:
                    label += " (fit unavailable)"
                panel_legend.append(Line2D([], [], color=COLORS[genotype], marker=MARKERS[genotype],
                    linestyle="-" if available else "None", label=label, markersize=5))
            ax.legend(handles=panel_legend, loc="lower right", fontsize=8.5, framealpha=.9, borderpad=.5, handlelength=2)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
            ax.set_xticks([-2, -1, 1, 2], minor=True)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_xlabel("ILD (dB)", fontsize=11)
            ax.set_ylabel("P(right)", fontsize=11)
            ax.grid(alpha=.16)
            ax.spines[["top", "right"]].set_visible(False)
            assert len(ax.collections) == 3 and len(ax.lines) == valid_fits
            assert tuple(ax.get_xlim()) == XLIM and tuple(ax.get_ylim()) == YLIM
            panel_audits.append({"group": group, "scatter_points": 30, "valid_fit_curves": valid_fits})
        fig.subplots_adjust(left=.07 if view != "genotype" else .09, right=.985,
                            top=.81 if view == "filename_genotype" else .73,
                            bottom=.12 if view == "filename_genotype" else .20, wspace=.16, hspace=.36)
        weighting = "ABL trials pooled at each ILD: right / all choices." if method == "trial_pooled" else "Equal mean of measured ABL 20/40/60 P(right) at each ILD; average first, then fit."
        fig.text(.5, .015,
            "Training 16; confirmed repeat rule; 807 short-sound rows removed in preparation; all 53 animals retained.\n"
            "Correct/error choices; no RT restriction. ABL 59 → 60; signed ILDs ±1/2/4/8/16. "+weighting+"\n"
            "Unweighted four-parameter fits to ten ILD points. Colors identify genotype. n is the recorded eligible choice count.",
            fontsize=8.5, ha="center", va="bottom")
        prefix = f"ASD_session_type_1_{view}_{method}_genotype_overlay_psychometrics"
        png, pdf = OUTPUT/f"{prefix}.png", OUTPUT/f"{prefix}.pdf"
        fig.savefig(png, dpi=PNG_DPI)
        fig.savefig(pdf)
        plt.close(fig)
        with fitz.open(pdf) as document:
            assert document.page_count == 1 and method_config["title"] in document[0].get_text()
        pages.append({"method": method, "view": view, "panels": len(config["groups"]), "grid": config["shape"],
                      "scatter_points": sum(p["scatter_points"] for p in panel_audits),
                      "valid_fit_curves": sum(p["valid_fit_curves"] for p in panel_audits), "panel_audits": panel_audits,
                      "xlim": XLIM, "ylim": YLIM, "png": str(png.relative_to(REPO)), "pdf": str(pdf.relative_to(REPO)), "pdf_pages": 1})


# %% Validation manifest and unchanged-file hashes
for name, digest in protected_hashes.items():
    with (REPO/name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
assert len(pages) == 6 and sum(page["scatter_points"] for page in pages) == 420
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_hash, "prepared_rows": len(data),
    "observed_counts_per_view_and_method": observed, "included_nonfinite_timed_rt_choices": int((~np.isfinite(selected.timed_rt)).sum()),
    "selection": reference["selection"], "short_sound_cleanup": reference["short_sound_cleanup"],
    "ABLs": ABLS, "ABL_mapping": {"59": 60}, "signed_ILDS": ILDS,
    "genotype_order": GENOTYPES, "genotype_colors": {g: to_hex(c) for g, c in COLORS.items()}, "genotype_markers": MARKERS,
    "methods": METHODS, "pooling_within_ABL": "Observed choice counts across animals and sessions; no equal-animal averaging",
    "counts_in_averaged_point_audit": "Raw eligible choice totals are descriptive; equal-ABL P(right) uses three equal weights, not right_trials/trials",
    "model": "lower + (upper-lower)/(1+exp(-k*(ILD-x0)))", "parameter_order": ["upper", "lower", "x0", "k"],
    "fit_bounds": {"upper": [0, 1], "lower": [0, 1], "x0": "unrestricted", "k": [0, "infinity"]},
    "initial_guess": "[max(P), min(P), median(ILD), 0.1]", "maxfev": MAXFEV,
    "fit_weighting": "Unweighted curve_fit on ten combined signed-ILD probabilities, sigma=None",
    "curve_ILDS": CURVE_ILDS.tolist(), "fits_attempted": len(fits), "fits_per_method": fits.groupby("method").size().to_dict(),
    "abl_condition_points": len(abl_counts), "combined_points": len(points), "membership_records": len(membership),
    "fit_status_counts": status.to_dict(orient="records"), "pages": pages,
    "all_existing_psychometric_counts_and_memberships_match": True,
    "synthetic_checks": ["unequal-trial pooling differs from equal-ABL averaging", "recover known sigmoid parameters",
                         "optimizer failure audited", "non-increasing and zero-k fits audited"],
    "protected_file_sha256_unchanged": protected_hashes,
}
(OUTPUT/"overlay_validation.json").write_text(json.dumps(manifest, indent=2)+"\n")
print("Saved six genotype-overlay PNG/PDF pairs; all input CSVs and previous analyses are unchanged.", flush=True)
