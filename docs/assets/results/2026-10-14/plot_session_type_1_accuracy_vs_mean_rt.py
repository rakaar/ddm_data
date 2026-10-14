# %% Imports
"""One animal/ABL per dot: valid-choice accuracy versus mean finite RT.

Run .venv/bin/python asd_analysis/plot_session_type_1_accuracy_vs_mean_rt.py.
Accuracy keeps every selected valid choice, including those with missing RT;
mean RT uses every finite timed_rt, with no cutoff. Both denominators are
recorded. Pool observed trials across signed ILDs and source sessions.
"""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_accuracy_vs_mean_rt"
RT_REFERENCE = REPO / "asd_analysis/session_type_1_animal_rt_summary"
PSYCHOMETRIC_REFERENCE = REPO / "asd_analysis/session_type_1_slope_jnd/slope_jnd_condition_counts.csv"
ABLS = [20, 40, 60]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
MARKERS = {"wt": "o", "het": "s", "hom": "^"}
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
FAMILY_FILL = {"SHANK3": "hollow", "CNTNAP2": "filled"}
FIGSIZE = (9.5, 7.5)
PNG_DPI = 200
STEM = "ASD_session_type_1_accuracy_vs_mean_rt"
OUTPUT.mkdir(parents=True, exist_ok=True)
input_stat = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
protected = [INPUT, PSYCHOMETRIC_REFERENCE, RT_REFERENCE / "rt_statistics.csv"]
protected.extend(sorted(RT_REFERENCE.glob("*.png")))
protected.extend(sorted(RT_REFERENCE.glob("*.pdf")))
protected_hashes = {}
for path in protected:
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
reference_manifest = json.loads((RT_REFERENCE / "validation.json").read_text())
assert reference_manifest["input_sha256"] == input_sha256


# %% Same valid-choice selection; the prepared CSV already applies the repeat rule
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "ABL", "ILD", "success", "timed_rt"], low_memory=False)
assert len(data) == 642370 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
data["plot_ABL"] = data.ABL.replace({59: 60})
choices = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
choices["plot_ABL"] = choices.plot_ABL.astype(int)
choices["correct"] = choices.success.eq(1)
choices["finite_rt"] = np.isfinite(choices.timed_rt)
finite = choices.loc[choices.finite_rt].copy()
assert len(choices) == 471416 and len(finite) == 469228 and choices.animal.nunique() == 53
print(f"{len(choices):,} valid choices; {len(finite):,} finite RTs; {choices.animal.nunique()} animals", flush=True)


# %% One pooled accuracy and RT mean per animal/ABL, with both denominators
keys = ["filename", "genotype", "animal", "plot_ABL"]
points = choices.groupby(keys, as_index=False).agg(
    valid_choice_trials=("success", "size"), correct_trials=("correct", "sum"),
    finite_rt_trials=("finite_rt", "sum"))
points["error_trials"] = points.valid_choice_trials - points.correct_trials
points["accuracy"] = points.correct_trials / points.valid_choice_trials
points["accuracy_percent"] = 100 * points.accuracy
points["nonfinite_rt_trials"] = points.valid_choice_trials - points.finite_rt_trials
rt_stats = finite.groupby(keys, as_index=False).agg(
    mean_rt_s=("timed_rt", "mean"), correct_finite_rt_trials=("correct", "sum"),
    rt_above_1_s=("timed_rt", lambda values: values.gt(1).sum()))
points = points.merge(rt_stats, on=keys, validate="one_to_one")
# This additional accuracy is an audit only; the scatter uses all valid choices.
points["error_finite_rt_trials"] = points.finite_rt_trials - points.correct_finite_rt_trials
points["accuracy_finite_rt_only"] = points.correct_finite_rt_trials / points.finite_rt_trials
points["accuracy_difference_from_finite_only"] = points.accuracy - points.accuracy_finite_rt_only
points["color"] = points.plot_ABL.map({a: to_hex(c) for a, c in COLORS.items()})
points["family"] = points.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
points["marker"] = points.genotype.map(MARKERS)
points["fill_style"] = points.family.map(FAMILY_FILL)
assert points[["family", "marker", "fill_style"]].notna().all().all()
assert len(points) == 159 and points.animal.nunique() == 53
assert points.groupby("plot_ABL").animal.nunique().to_dict() == {20: 53, 40: 53, 60: 53}
assert points.accuracy.between(0, 1).all() and np.isfinite(points.mean_rt_s).all()
assert points.valid_choice_trials.sum() == 471416 and points.finite_rt_trials.sum() == 469228
assert points.nonfinite_rt_trials.sum() == 2188 and points.rt_above_1_s.sum() == 2250
assert set(points.loc[points.nonfinite_rt_trials.gt(0), "animal"]) == {"ASD0053"}


# %% Existing psychometric counts and RT means must match exactly
old_rt = pd.read_csv(RT_REFERENCE / "rt_statistics.csv").set_index(keys).sort_index()
new_rt = points.set_index(keys).sort_index()
np.testing.assert_allclose(new_rt.mean_rt_s, old_rt.mean_rt_s, rtol=0, atol=1e-12)
np.testing.assert_array_equal(new_rt.finite_rt_trials, old_rt.finite_rt_trials)
np.testing.assert_array_equal(new_rt.valid_choice_trials, old_rt.choice_trials)
old_choices = pd.read_csv(PSYCHOMETRIC_REFERENCE)
old_choices = old_choices.loc[old_choices.view.eq("animal")].copy()
old_choices[["filename", "animal"]] = old_choices.group.str.rsplit("/", n=1, expand=True)
old_choices["plot_ABL"] = old_choices.plot_ABL.astype(int)
old_choices = old_choices.groupby(keys)[["trials", "correct_trials", "error_trials"]].sum().sort_index()
np.testing.assert_array_equal(new_rt.valid_choice_trials, old_choices.trials)
np.testing.assert_array_equal(new_rt.correct_trials, old_choices.correct_trials)
np.testing.assert_array_equal(new_rt.error_trials, old_choices.error_trials)
points.to_csv(OUTPUT / "animal_abl_points.csv", index=False)


# %% A single scatter: ABL-colored dots, with no fits or error bars
fig, ax = plt.subplots(figsize=FIGSIZE)
fig.suptitle("Session type 1 | Accuracy versus mean RT", fontsize=17, y=.975)
fig.text(.5, .925, "53 animals | 159 animal–ABL dots | signed ILDs pooled", ha="center", fontsize=12)
for abl in ABLS:
    # Draw filled points first so hollow outlines remain visible in dense areas.
    for family in ["CNTNAP2", "SHANK3"]:
        for genotype in MARKERS:
            condition = points.loc[points.plot_ABL.eq(abl) & points.family.eq(family) & points.genotype.eq(genotype)]
            filled = FAMILY_FILL[family] == "filled"
            scatter = ax.scatter(condition.mean_rt_s, condition.accuracy_percent, s=58,
                                 marker=MARKERS[genotype], facecolors=COLORS[abl] if filled else "none",
                                 edgecolors=COLORS[abl], linewidths=.9 if filled else 1.4, alpha=.95,
                                 zorder=3 if filled else 4)
            np.testing.assert_allclose(scatter.get_offsets(), condition[["mean_rt_s", "accuracy_percent"]].to_numpy())
            assert to_hex(scatter.get_edgecolors()[0]) == to_hex(COLORS[abl])
            if filled:
                assert to_hex(scatter.get_facecolors()[0]) == to_hex(COLORS[abl])
            else:
                assert len(scatter.get_facecolors()) == 0
assert len(ax.collections) == 18 and sum(len(c.get_offsets()) for c in ax.collections) == 159
ax.set_xlabel("Mean measured RT (s)", fontsize=13)
ax.set_ylabel("Accuracy (% correct among valid choices)", fontsize=13)
ax.set_xlim(0, float(points.mean_rt_s.max()) * 1.12)
ax.set_ylim(min(50, float(np.floor((points.accuracy_percent.min() - 2) / 5) * 5)), 100)
ax.grid(alpha=.18)
ax.spines[["top", "right"]].set_visible(False)
color_legend = ax.legend(handles=[Line2D([], [], marker="o", color=COLORS[a], linestyle="None",
    markersize=6, label=f"{a} dB") for a in ABLS], title="ABL (color)", loc="lower left",
    frameon=False, fontsize=10, title_fontsize=10)
ax.add_artist(color_legend)
shape_legend = ax.legend(handles=[Line2D([], [], marker=MARKERS[g], color=".25", linestyle="None",
    markersize=6, label=GENOTYPE_LABELS[g]) for g in MARKERS], title="Genotype (shape)", loc="lower center",
    frameon=False, fontsize=10, title_fontsize=10)
ax.add_artist(shape_legend)
ax.legend(handles=[Line2D([], [], marker="o", color=".25", linestyle="None", markersize=6,
    markerfacecolor="none" if FAMILY_FILL[f] == "hollow" else ".25", markeredgewidth=1.3,
    label=f"{f} ({FAMILY_FILL[f]})") for f in FAMILY_FILL], title="Gene family (fill)", loc="lower right",
    frameon=False, fontsize=10, title_fontsize=10)
fig.text(.5, .084, "Valid trials: success in {1,-1}; accuracy = correct / (correct + error).\n"
         "Mean RT uses all finite timed_rt values, including >1 s; no RT cutoff.", ha="center", fontsize=9)
fig.text(.5, .034, "ABL 59 mapped to 60; signed ILDs ±1/2/4/8/16 dB and sessions pooled by observed trials.\n"
         "Accuracy: 471,416 choices; mean RT: 469,228 finite RTs. Missing RTs stay in accuracy only.", ha="center", fontsize=9)
fig.subplots_adjust(left=.12, right=.97, top=.875, bottom=.21)
png, pdf = OUTPUT / f"{STEM}.png", OUTPUT / f"{STEM}.pdf"
fig.savefig(png, dpi=PNG_DPI, facecolor="white")
fig.savefig(pdf, facecolor="white")
plt.close(fig)


# %% Saved-PDF checks, counts, and provenance
with fitz.open(pdf) as document:
    assert document.page_count == 1
    paths = document[0].get_drawings()
    for abl, color in COLORS.items():
        rgb = matplotlib.colors.to_rgb(color)
        filled = sum(p["fill"] is not None and np.allclose(p["fill"], rgb, atol=1e-6) for p in paths)
        hollow = sum(p["fill"] is None and p["color"] is not None and np.allclose(p["color"], rgb, atol=1e-6) for p in paths)
        assert filled == 34 and hollow == 20, (abl, filled, hollow)  # 33 CNTNAP2 + color legend; 20 SHANK3.
assert input_stat == (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
manifest = {"status": "passed", "input": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256,
    "source_script": str(Path(__file__).relative_to(REPO)), "animals": 53, "animal_abl_points": 159,
    "valid_choice_trials": len(choices), "correct_trials": int(choices.correct.sum()),
    "error_trials": int((~choices.correct).sum()), "finite_rt_trials": len(finite),
    "nonfinite_rt_trials": len(choices) - len(finite), "rt_above_1_s": int(finite.timed_rt.gt(1).sum()),
    "selection": "Cleaned session-type-1 CSV; training 16; success in {1,-1}; ABL 20/40/60 after 59 to 60; signed ILDs +/-1/2/4/8/16; no RT cutoff",
    "repeat_filter_reapplied": False, "accuracy_formula": "correct valid choices / all valid choices; include missing RTs",
    "mean_rt_formula": "Arithmetic mean of all finite valid-trial timed_rt values (seconds)",
    "pooling": "Observed raw trials across signed ILDs and sessions, within animal and ABL",
    "x_axis": "mean_rt_s", "y_axis": "accuracy_percent", "abl_colors": {str(a): to_hex(c) for a, c in COLORS.items()},
    "genotype_markers": MARKERS, "family_fill_styles": FAMILY_FILL,
    "points_by_family_and_abl": points.groupby(["family", "plot_ABL"]).size().rename("points").reset_index().to_dict(orient="records"),
    "points_by_genotype_and_abl": points.groupby(["genotype", "plot_ABL"]).size().rename("points").reset_index().to_dict(orient="records"),
    "different_denominator_points": points.loc[points.nonfinite_rt_trials.gt(0),
        ["animal", "plot_ABL", "valid_choice_trials", "finite_rt_trials", "nonfinite_rt_trials", "accuracy", "accuracy_finite_rt_only"]].to_dict(orient="records"),
    "png": str(png.relative_to(REPO)), "pdf": str(pdf.relative_to(REPO)), "pdf_pages": 1,
    "protected_file_sha256": protected_hashes,
    "checks": ["159 animal/ABL dots; 53 animals at each ABL", "accuracy counts match existing psychometric audits",
               "RT means/counts match existing all-finite-RT summaries", "plotted coordinates and ABL colors",
               "SHANK3 hollow; CNTNAP2 filled; WT circle, HET square, HOM triangle",
               "single-page PDF with 33 filled and 20 hollow data markers per ABL", "input size and modification time unchanged"]}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved {png}\nSaved {pdf}", flush=True)
print("Different denominator points:", json.dumps(manifest["different_denominator_points"]), flush=True)
