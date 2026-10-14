# %% Imports
"""Three trial-pooled genotype psychometrics with measured timed_rt in [0,1] s.

The task-setting column max_rt is NOT filtered: it equals 10 in this dataset.
Run with .venv/bin/python; preserve the preceding unrestricted/averaged figures.
"""
import hashlib
import json
from pathlib import Path
import warnings

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
INPUT = REPO/"asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO/"asd_analysis/session_type_1_genotype_overlay_psychometrics_rt_0_to_1"
RTD_REFERENCE = REPO/"asd_analysis/session_type_1_ild_pooled_rtds"
RT_WINDOW = (0, 1)  # Inclusive measured reaction times in seconds; not max_rt.
SOURCE_FILES = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
GENOTYPES = ["wt", "het", "hom"]
LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
COLORS = {"wt": "tab:blue", "het": "tab:orange", "hom": "tab:green"}
MARKERS = {"wt": "o", "het": "s", "hom": "^"}
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
XLIM, YLIM = (-18, 18), (0, 1)
CURVE_ILDS = np.linspace(*XLIM, 501)
PNG_DPI = 180
EXPECTED = {"animals": 53, "trials": 466978, "right": 239331, "left": 227647}
VIEWS = {
    "filename_genotype": {"column": "filename", "groups": SOURCE_FILES, "shape": (2, 2), "figsize": (14, 10), "title": "Four cohorts separately", "curves": 12},
    "family_genotype": {"column": "family", "groups": ["SHANK3", "CNTNAP2"], "shape": (1, 2), "figsize": (16, 6), "title": "SHANK3 and CNTNAP2 separately", "curves": 6},
    "genotype": {"column": None, "groups": ["all_cohorts"], "shape": (1, 1), "figsize": (12, 7), "title": "Both families and all cohorts combined", "curves": 3},
}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Preserve all input CSVs and preceding analysis/figure exports
protected = [REPO/name for name in SOURCE_FILES] + [REPO/"metadata.csv"]
protected += list(INPUT.parent.rglob("*.csv")) + list(INPUT.parent.rglob("*.json"))
for folder in ["session_type_1_rtds", "session_type_1_genotype_rtds", "session_type_1_ild_pooled_rtds",
               "session_type_1_psychometrics", "session_type_1_slope_jnd", "session_type_1_genotype_overlay_psychometrics"]:
    protected += [p for p in (REPO/"asd_analysis"/folder).iterdir() if p.is_file()]
for date in ["2026-10-12", "2026-10-13", "2026-10-14"]:
    protected += list((REPO/"docs/assets/results"/date).iterdir())
for date in ["2026-10-12", "2026-10-13"]:
    protected += [REPO/"docs/results"/f"{date}.md"]
protected_hashes = {}
for path in set(protected):
    if path.is_file():
        with path.open("rb") as stream:
            protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_hash = protected_hashes[str(INPUT.relative_to(REPO))]
reference = json.loads((RTD_REFERENCE/"ild_pooled_rtd_validation.json").read_text())
assert reference["status"] == "passed" and reference["input_sha256"] == input_hash


# %% Select the same choices, then impose the confirmed measured-RT window
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "short_duration", "ABL", "ILD", "success", "response_poke", "timed_rt", "max_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all() and data.trial_is_repeat.eq(False).all()
assert data.is_short_sound.eq(0).all() and data.short_duration.eq(0).all()
assert data.max_rt.eq(10).all()  # Document the setting; it is not the selection variable.
data["plot_ABL"] = data.ABL.replace({59: 60})
base = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(ILDS)].copy()
assert len(base) == 471416 and base.animal.nunique() == 53
finite = np.isfinite(base.timed_rt)
in_window = finite & base.timed_rt.between(*RT_WINDOW, inclusive="both")
excluded = {"nonfinite_rt": int((~finite).sum()), "below_window": int((finite & base.timed_rt.lt(RT_WINDOW[0])).sum()),
            "above_window": int((finite & base.timed_rt.gt(RT_WINDOW[1])).sum())}
assert excluded == {"nonfinite_rt": 2188, "below_window": 0, "above_window": 2250}
selected = base.loc[in_window].copy()
selected["plot_ABL"] = selected.plot_ABL.astype(int)
assert selected.response_poke.isin([1, -1]).all() and selected.genotype.isin(GENOTYPES).all()
selected["right"] = selected.response_poke.eq(1)
selected["correct"] = selected.success.eq(1)
selected["family"] = selected.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
observed = {"animals": selected.animal.nunique(), "trials": len(selected),
            "right": int(selected.right.sum()), "left": int((~selected.right).sum())}
assert observed == EXPECTED and set(selected.animal) == set(base.animal)
assert {"ASD0026", "ASD0045"} <= set(selected.animal)
cohort2_animals = sorted(selected.loc[selected.filename.eq("SHANK3_cohort2.csv"), "animal"].unique())
assert len(cohort2_animals) == 6
print("Measured-RT selection:", json.dumps(observed), "Excluded:", excluded, flush=True)


# %% Pool raw counts across ABLs at each signed ILD; retain ABL/group/membership audits
abl_parts, point_parts, group_parts, member_parts = [], [], [], []
for view, config in VIEWS.items():
    rows = selected.copy()
    rows["group"] = rows[config["column"]] if config["column"] else "all_cohorts"
    for columns, destination in [(["group", "genotype", "plot_ABL", "ILD"], abl_parts),
                                 (["group", "genotype", "ILD"], point_parts), (["group", "genotype"], group_parts)]:
        counts = rows.groupby(columns, as_index=False).agg(trials=("response_poke", "size"), right_trials=("right", "sum"),
            correct_trials=("correct", "sum"), animals=("animal", "nunique"))
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
points = pd.concat(point_parts, ignore_index=True)
groups = pd.concat(group_parts, ignore_index=True)
membership = pd.concat(member_parts, ignore_index=True)
points["p_right"] = points.right_trials/points.trials
assert len(abl_counts) == 630 and len(points) == 210 and len(groups) == 21 and len(membership) == 159
keys = ["view", "group", "genotype", "ILD"]
fields = ["trials", "right_trials", "left_trials", "correct_trials", "error_trials"]
pd.testing.assert_frame_equal(abl_counts.groupby(keys)[fields].sum().sort_index(), points.set_index(keys)[fields].sort_index())
for view in VIEWS:
    assert groups.loc[groups.view.eq(view), ["trials", "right_trials", "left_trials", "animals"]].sum().tolist() == [466978, 239331, 227647, 53]
# These choices must match the independently verified all-ILD RTD normalization samples.
old_rtd = pd.read_csv(RTD_REFERENCE/"ild_pooled_rtd_curve_counts.csv")
rtd_keys = ["view", "group", "genotype", "plot_ABL"]
new_rtd_n = abl_counts.groupby(rtd_keys).trials.sum().sort_index()
pd.testing.assert_series_equal(new_rtd_n, old_rtd.set_index(rtd_keys).window_trials.sort_index(), check_names=False)
abl_counts.to_csv(OUTPUT/"rt_window_abl_counts.csv", index=False)
groups.to_csv(OUTPUT/"rt_window_group_counts.csv", index=False)
membership.to_csv(OUTPUT/"rt_window_membership.csv", index=False)


# %% Unweighted four-parameter fits to ten pooled ILD probability points
def sigmoid(x, upper, lower, x0, k):
    return lower+(upper-lower)/(1+np.exp(-k*(x-x0)))


fit_rows = []
points["fitted_p_right"] = np.nan
for fit_keys, condition in points.groupby(["view", "group", "genotype"], sort=False):
    condition = condition.sort_values("ILD")
    assert condition.ILD.tolist() == ILDS
    x, y = condition.ILD.to_numpy(dtype=float), condition.p_right.to_numpy()
    initial = [float(y.max()), float(y.min()), float(np.median(x)), INITIAL_K]
    result = {"upper": np.nan, "lower": np.nan, "x0": np.nan, "k": np.nan, "fit_status": "optimizer_failed",
        "initial_parameters": json.dumps(initial), "optimizer_code": 0, "optimizer_message": "", "function_evaluations": 0,
        "warnings": "", "covariance_finite": False, "unweighted_sse": np.nan, "rmse": np.nan}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, code = curve_fit(sigmoid, x, y, p0=initial, bounds=FIT_BOUNDS, maxfev=MAXFEV, full_output=True)
        result.update(zip(["upper", "lower", "x0", "k"], parameters))
        result.update(optimizer_code=int(code), optimizer_message=message, function_evaluations=int(info["nfev"]),
            covariance_finite=bool(np.isfinite(covariance).all()), warnings="; ".join(dict.fromkeys(str(w.message) for w in caught)))
        if np.isfinite(parameters).all() and parameters[0] > parameters[1] and parameters[3] > 0:
            prediction = sigmoid(x, *parameters)
            residual = prediction-y
            result.update(fit_status="ok", unweighted_sse=float(residual @ residual), rmse=float(np.sqrt(np.mean(residual**2))))
            points.loc[condition.index, "fitted_p_right"] = prediction
        else:
            result["fit_status"] = "non_increasing_or_nonfinite"
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["optimizer_message"] = str(error)
    view, group, genotype = fit_keys
    group_info = groups.loc[groups.view.eq(view) & groups.group.eq(group) & groups.genotype.eq(genotype)].iloc[0]
    fit_rows.append({"view": view, "group": group, "genotype": genotype, "animals": int(group_info.animals),
        "trials": int(group_info.trials), "ILD_points": len(condition), **result})
fits = pd.DataFrame(fit_rows)
points["residual"] = points.fitted_p_right-points.p_right
assert len(fits) == 21 and fits.groupby("view").size().to_dict() == {view: cfg["curves"] for view, cfg in VIEWS.items()}
points.to_csv(OUTPUT/"rt_window_points.csv", index=False)
fits.to_csv(OUTPUT/"rt_window_fits.csv", index=False)
status = fits.groupby(["view", "fit_status"]).size().rename("count").reset_index()
status.to_csv(OUTPUT/"rt_window_fit_status_counts.csv", index=False)
print(status.to_string(index=False), flush=True)


# %% Three genotype-overlay figures, trial-pooled only
pages = []
legend = [Line2D([], [], color=COLORS[g], marker=MARKERS[g], linestyle="-", label=LABELS[g], markersize=6) for g in GENOTYPES]
for view, config in VIEWS.items():
    fig, axes = plt.subplots(*config["shape"], figsize=config["figsize"], sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"Session type 1 | Genotype psychometrics\n{config['title']}\nTrials pooled across ABLs | measured RT in [0,1] s", fontsize=15, y=.985)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .905 if view == "filename_genotype" else .87), ncol=3, frameon=False)
    panels = []
    for group, ax in zip(config["groups"], axes.flat):
        panel = groups.loc[groups.view.eq(view) & groups.group.eq(group)]
        label = group.removesuffix(".csv") if group != "all_cohorts" else "All four cohorts"
        ax.set_title(f"{label}\n{int(panel.animals.sum())} animals | {int(panel.trials.sum()):,} RT-window choices", fontsize=11, pad=10)
        handles, valid_fits = [], 0
        for genotype in GENOTYPES:
            condition = points.loc[points.view.eq(view) & points.group.eq(group) & points.genotype.eq(genotype)].sort_values("ILD")
            estimate = fits.loc[fits.view.eq(view) & fits.group.eq(group) & fits.genotype.eq(genotype)].iloc[0]
            scatter = ax.scatter(condition.ILD, condition.p_right, color=COLORS[genotype], marker=MARKERS[genotype], s=33, alpha=.9, zorder=3)
            np.testing.assert_allclose(scatter.get_offsets(), condition[["ILD", "p_right"]].to_numpy())
            assert to_hex(scatter.get_facecolors()[0]) == to_hex(COLORS[genotype])
            available = estimate.fit_status == "ok"
            if available:
                curve = sigmoid(CURVE_ILDS, *[estimate[key] for key in ["upper", "lower", "x0", "k"]])
                assert np.isfinite(curve).all() and ((curve >= 0) & (curve <= 1)).all()
                ax.plot(CURVE_ILDS, curve, color=COLORS[genotype], linestyle="-", linewidth=1.8, zorder=2)
                valid_fits += 1
            word = "animal" if estimate.animals == 1 else "animals"
            label = f"{LABELS[genotype]}: {int(estimate.animals)} {word}, n={int(estimate.trials):,}"
            if not available:
                label += " (fit unavailable)"
            handles.append(Line2D([], [], color=COLORS[genotype], marker=MARKERS[genotype], linestyle="-" if available else "None", label=label, markersize=5))
        ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=.9, borderpad=.5, handlelength=2)
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
        panels.append({"group": group, "scatter_points": 30, "valid_fit_curves": valid_fits})
    fig.subplots_adjust(left=.09 if view == "genotype" else .07, right=.985,
        top=.81 if view == "filename_genotype" else .73, bottom=.12 if view == "filename_genotype" else .20, wspace=.16, hspace=.36)
    fig.text(.5, .015,
        "Training 16; confirmed repeat rule; 807 short-sound rows removed in preparation; all 53 animals retained.\n"
        "Correct/error choices; finite measured timed_rt in inclusive [0,1] s. ABL 59 → 60; signed ILDs ±1/2/4/8/16.\n"
        "Pool raw ABL counts at each ILD, then fit ten probabilities equally. Colors identify genotype; n counts RT-window choices.",
        fontsize=8.5, ha="center", va="bottom")
    prefix = f"ASD_session_type_1_{view}_trial_pooled_rt_0_to_1_genotype_overlay_psychometrics"
    png, pdf = OUTPUT/f"{prefix}.png", OUTPUT/f"{prefix}.pdf"
    fig.savefig(png, dpi=PNG_DPI)
    fig.savefig(pdf)
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert document.page_count == 1 and "measured RT in [0,1] s" in document[0].get_text()
    pages.append({"view": view, "panels": len(config["groups"]), "grid": config["shape"], "panel_audits": panels,
        "scatter_points": sum(p["scatter_points"] for p in panels), "valid_fit_curves": sum(p["valid_fit_curves"] for p in panels),
        "xlim": XLIM, "ylim": YLIM, "pdf_pages": 1, "png": str(png.relative_to(REPO)), "pdf": str(pdf.relative_to(REPO))})


# %% Confirm preservation and write the selection/fit/figure manifest
for name, digest in protected_hashes.items():
    with (REPO/name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
manifest = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_hash, "prepared_rows": len(data),
    "base_choice_trials": len(base), "observed_counts_per_view": observed, "RT_exclusions": excluded,
    "selection": "session_type=1; training_level=16; prepared next-trial repeat rule with CNP/Fixation exceptions; success in {1,-1}; ABL 59->60, 20/40/60; signed ILDs +/-1/2/4/8/16; finite recorded timed_rt in inclusive [0,1] seconds",
    "RT_field": "timed_rt", "RT_window_seconds_inclusive": RT_WINDOW, "max_rt_setting_filtered": False,
    "max_rt_setting_values": sorted(base.max_rt.unique().tolist()), "literal_max_rt_equals_1_choices": int(base.max_rt.eq(1).sum()),
    "short_sound_removed_during_preparation": 807, "additional_animal_exclusions": [], "SHANK3_cohort2_animals": cohort2_animals,
    "pooling": "Raw right/left trial counts pooled across all three ABLs and animals at each signed ILD; no ABL probability or animal averages",
    "ABLs": ABLS, "signed_ILDS": ILDS, "genotype_colors": {g: to_hex(c) for g, c in COLORS.items()}, "genotype_markers": MARKERS,
    "parameter_order": ["upper", "lower", "x0", "k"], "initial_guess": "[max(P), min(P), median(ILD), 0.1]",
    "fit_bounds": {"upper": [0, 1], "lower": [0, 1], "x0": "unrestricted", "k": [0, "infinity"]}, "maxfev": MAXFEV,
    "fit_weighting": "Unweighted curve_fit on ten pooled ILD probabilities; sigma=None", "fit_status_counts": status.to_dict(orient="records"),
    "fits_attempted": len(fits), "condition_points": len(points), "ABL_condition_points": len(abl_counts), "membership_records": len(membership),
    "all_ABL_trial_counts_match_existing_RTD_window_counts": True, "pages": pages, "protected_file_sha256_unchanged": protected_hashes,
}
(OUTPUT/"rt_window_validation.json").write_text(json.dumps(manifest, indent=2)+"\n")
print("Saved three RT-restricted trial-pooled PNG/PDF pairs; all previous analysis exports and CSVs remain unchanged.", flush=True)
