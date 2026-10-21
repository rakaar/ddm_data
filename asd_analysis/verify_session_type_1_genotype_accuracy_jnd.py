# %% Imports and independent parameters
"""Reconstruct selections/statistics from CSV and inspect saved PDF marks."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import warnings
from unittest.mock import Mock

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq, curve_fit
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_accuracy_jnd"
manifest = json.loads((OUTPUT / "validation.json").read_text())
ABLS, ILDS = [20, 40, 60], [1, 2, 4, 8, 16]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
GENOTYPES = ["wt", "hom", "het"]
COHORTS = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
COLORS = {"wt": "#1f77b4", "hom": "#2ca02c", "het": "#ff7f0e"}
OFFSETS = {"wt": -.12, "hom": 0., "het": .12}
assert manifest["status"] == "passed" and not manifest["repeat_filter_reapplied"]


# %% Independent standard-library CSV selection and signed-condition counts
signed = defaultdict(Counter)
identity = {}
prepared_rows = 0
with (REPO / manifest["input_csv"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        prepared_rows += 1
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"] or 0) == 0
        assert not (row["filename"] == "CNTNAP2_cohort4.csv" and float(row["session"]) < 13)
        if float(row["success"] or "nan") not in {1, -1}:
            continue
        rt, abl, ild = [float(row[k] or "nan") for k in ["timed_rt", "ABL", "ILD"]]
        abl = 60 if abl == 59 else abl
        if not math.isfinite(rt) or not 0 <= rt <= 1 or abl not in ABLS or ild not in SIGNED_ILDS:
            continue
        animal = row["animal"]
        ids = (row["filename"], row["filename"].split("_")[0], row["genotype"])
        if animal in identity:
            assert identity[animal] == ids
        identity[animal] = ids
        response = float(row["response_poke"])
        assert response in {1, -1}
        count = signed[(animal, int(abl), int(ild))]
        count.update(n=1, correct=int(float(row["success"]) == 1), right=int(response == 1))
assert prepared_rows == 610603 and len(identity) == 53 and len(signed) == 1590
assert sum(v["n"] for v in signed.values()) == 446949
assert sum(v["correct"] for v in signed.values()) == 343217
assert sum(v["right"] for v in signed.values()) == 229058
expected_conditions = defaultdict(Counter)
condition_animals = defaultdict(set)
member_counts = Counter()
for (animal, abl, ild), counts in signed.items():
    filename, family, genotype = identity[animal]
    for view, group in [("animal", animal), ("cohort", filename), ("family", family)]:
        key = (view, group, genotype, abl, ild)
        expected_conditions[key].update(counts)
        condition_animals[key].add(animal)
        if view != "animal":
            member_counts[(view, group, filename, family, genotype, animal)] += counts["n"]
conditions = pd.read_csv(OUTPUT / "condition_counts.csv")
assert len(conditions) == len(expected_conditions) == 2130
for r in conditions.itertuples(index=False):
    key = (r.view, r.group, r.genotype, r.plot_ABL, r.ILD)
    e = expected_conditions[key]
    assert [r.choices, r.correct_choices, r.right_choices, r.left_choices, r.pool_animals] == [e["n"], e["correct"], e["right"], e["n"]-e["right"], len(condition_animals[key])]
    assert math.isclose(r.p_right, e["right"]/e["n"], rel_tol=0, abs_tol=2e-16)
assert conditions.groupby("view").choices.sum().to_dict() == {"animal":446949, "cohort":446949, "family":446949}


# %% Animal accuracies, pooled accuracies and independently centered animal SEMs
animal_expected = {}
for animal in identity:
    for abl in ABLS:
        for ild in ILDS:
            plus, minus = signed[(animal, abl, ild)], signed[(animal, abl, -ild)]
            n, correct = plus["n"]+minus["n"], plus["correct"]+minus["correct"]
            animal_expected[(animal, abl, ild)] = (n, correct, correct/n, plus["n"], minus["n"])
animal_audit = pd.read_csv(OUTPUT / "animal_accuracy.csv")
assert len(animal_audit) == 795
for r in animal_audit.itertuples(index=False):
    e = animal_expected[(r.animal, r.plot_ABL, r.abs_ILD)]
    assert [r.window_trials, r.correct_choices, r.positive_ild_trials, r.negative_ild_trials] == [e[0], e[1], e[3], e[4]]
    assert math.isclose(r.animal_accuracy, e[2], abs_tol=2e-16, rel_tol=0)
    assert (r.filename, r.family, r.genotype) == identity[r.animal]
accuracy = pd.read_csv(OUTPUT / "accuracy_points.csv")
expected_points = {}
for r in accuracy.itertuples(index=False):
    animals = [a for a, (filename, family, genotype) in identity.items()
        if genotype == r.genotype and (filename if r.view == "cohort" else family) == r.group]
    values = [animal_expected[(a, r.plot_ABL, r.abs_ILD)] for a in animals]
    n, correct = sum(v[0] for v in values), sum(v[1] for v in values)
    fractions = [v[2] for v in values]
    pooled = correct/n
    center = statistics.fmean(fractions)
    sd = statistics.stdev(fractions) if len(fractions) > 1 else math.nan
    sem = sd/math.sqrt(len(fractions))
    weighted = math.fsum(v[0]*v[2] for v in values)/n
    assert math.isclose(pooled, weighted, rel_tol=0, abs_tol=2e-16)
    assert [r.window_trials, r.correct_choices, r.pool_animals, r.sem_animals] == [n, correct, len(animals), len(animals)]
    assert r.positive_ild_trials == sum(v[3] for v in values) and r.negative_ild_trials == sum(v[4] for v in values)
    np.testing.assert_allclose([r.pooled_value, r.animal_estimate_mean, r.animal_estimate_sd, r.sem], [pooled, center, sd, sem], rtol=0, atol=3e-16, equal_nan=True)
    assert r.sem_status == ("ok" if len(animals) > 1 else "unavailable_single_animal")
    np.testing.assert_allclose([r.lower_bar,r.upper_bar], [pooled-sem,pooled+sem], rtol=0, atol=3e-16, equal_nan=True)
    key = ("accuracy", r.view, r.group, r.genotype, r.plot_ABL, r.abs_ILD)
    expected_points[key] = (pooled, sem, len(animals), len(animals), ILDS.index(r.abs_ILD)+OFFSETS[r.genotype])
assert len(accuracy) == 270 and accuracy["sem"].notna().sum() == 240
assert accuracy.groupby("view").window_trials.sum().to_dict() == {"cohort":446949, "family":446949}
missing = accuracy.loc[accuracy["sem"].isna()]
assert len(missing) == 30 and missing.view.eq("cohort").all() and missing.group.eq("SHANK3_cohort2.csv").all() and set(missing.genotype) == {"wt", "hom"}


# %% Independent refits and numerical root-finding for all available JND crossings
def independent_curve(x, upper, lower, x0, k):
    return lower+(upper-lower)*expit(k*(x-x0))


fits = pd.read_csv(OUTPUT / "jnd_fits.csv")
assert len(fits) == 213 and fits.groupby("view").size().to_dict() == {"animal":159, "cohort":36, "family":18}
roots_checked, fit_predictions = 0, {}
for r in fits.itertuples(index=False):
    c = conditions.loc[conditions.view.eq(r.view) & conditions.group.eq(r.group) & conditions.genotype.eq(r.genotype) & conditions.plot_ABL.eq(r.plot_ABL)].sort_values("ILD")
    x, probability = c.ILD.to_numpy(), c.p_right.to_numpy()
    assert x.tolist() == SIGNED_ILDS
    initial = [max(probability), min(probability), float(np.median(x)), .1]
    np.testing.assert_allclose(json.loads(r.initial_parameters), initial, atol=2e-16, rtol=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        independent, _ = curve_fit(independent_curve, x, probability, p0=initial,
            bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]), maxfev=10000)
    saved = np.array([r.upper, r.lower, r.x0, r.k])
    prediction = independent_curve(x, *saved)
    np.testing.assert_allclose(prediction, independent_curve(x, *independent), rtol=0, atol=2e-6)
    np.testing.assert_allclose(c.fitted_p_right, prediction, rtol=0, atol=5e-16)
    assert r.fit_status == "ok" and 0 <= r.lower < r.upper <= 1 and r.k > 0
    assert r.window_trials == c.choices.sum() and r.pool_animals == c.pool_animals.max()
    assert math.isclose(r.unweighted_sse, float(np.sum((prediction-probability)**2)), rel_tol=0, abs_tol=5e-16)
    low, high = .25*(1-r.lower), .75*r.upper
    np.testing.assert_allclose([r.y_low, r.y_high], [low, high], rtol=0, atol=2e-16)
    if r.lower < low < high < r.upper:
        roots = [brentq(lambda z: independent_curve(z, *saved)-target, r.x0-100/r.k, r.x0+100/r.k, xtol=1e-12) for target in [low, high]]
        value = (roots[1]-roots[0])/2
        assert r.jnd_status == "ok"
        np.testing.assert_allclose([r.ild_low, r.ild_high, r.jnd], [*roots, value], rtol=0, atol=3e-11)
        roots_checked += 2
    else:
        value = math.nan
        assert r.jnd_status == "thresholds_outside_asymptotes_or_reversed" and math.isnan(r.jnd)
    fit_predictions[(r.view, r.group, r.genotype, r.plot_ABL)] = value
assert roots_checked == 416 and fits.loc[fits.view.eq("animal"), "jnd"].notna().sum() == 154
undefined = fits.loc[fits.view.eq("animal") & fits.jnd.isna()]
assert set(zip(undefined.group, undefined.plot_ABL)) == {("ASD0035",20), ("ASD0058",20), ("ASD0044",40), ("ASD0048",40), ("ASD0048",60)}
animal_jnd = pd.read_csv(OUTPUT / "animal_jnd.csv")
assert len(animal_jnd) == 159
for r in animal_jnd.itertuples(index=False):
    assert (r.filename,r.family,r.genotype) == identity[r.animal]
    np.testing.assert_allclose(r.jnd, fit_predictions[("animal",r.animal,r.genotype,r.plot_ABL)], rtol=0, atol=3e-11, equal_nan=True)
pd.testing.assert_frame_equal(pd.read_csv(OUTPUT / "fit_status_counts.csv"),
    fits.groupby(["view","fit_status","jnd_status"]).size().rename("fits").reset_index(), check_dtype=False)
jnd = pd.read_csv(OUTPUT / "jnd_points.csv")
assert len(jnd) == 54 and jnd["sem"].notna().sum() == 47
for r in jnd.itertuples(index=False):
    animals = [a for a, (filename, family, genotype) in identity.items()
        if genotype == r.genotype and (filename if r.view == "cohort" else family) == r.group]
    values = [fit_predictions[("animal", a, r.genotype, r.plot_ABL)] for a in animals]
    values = [v for v in values if math.isfinite(v)]
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    sem = sd/math.sqrt(len(values)) if values else math.nan
    pooled = fit_predictions[(r.view, r.group, r.genotype, r.plot_ABL)]
    np.testing.assert_allclose([r.pooled_value, r.animal_estimate_mean, r.animal_estimate_sd, r.sem],
        [pooled, statistics.fmean(values), sd, sem], rtol=0, atol=2e-11, equal_nan=True)
    assert r.pool_animals == len(animals) and r.sem_animals == len(values) and r.unavailable_animal_jnds == len(animals)-len(values)
    assert r.sem_status == ("ok" if len(values) > 1 else "unavailable_single_animal")
    np.testing.assert_allclose([r.lower_bar,r.upper_bar], [pooled-sem,pooled+sem], rtol=0, atol=2e-11, equal_nan=True)
    expected_points[("jnd", r.view, r.group, r.genotype, r.plot_ABL, None)] = (pooled, sem, len(animals), len(values), GENOTYPES.index(r.genotype))
missing = jnd.loc[jnd["sem"].isna()]
assert len(missing) == 7 and missing.view.eq("cohort").all()
assert set(zip(missing.group, missing.genotype, missing.plot_ABL)) == {("SHANK3_cohort2.csv", g, a) for g in ["wt", "hom"] for a in ABLS} | {("CNTNAP2_cohort3.csv", "het", 20)}


# %% Exercise actual fit/JND helpers without executing the plotting script
source = REPO / manifest["source_script"]
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)]
context = {"np":np, "warnings":warnings, "json":json, "curve_fit":curve_fit,
    "FIT_BOUNDS":([0,0,-np.inf,0],[1,1,np.inf,np.inf]), "INITIAL_K":.1, "MAXFEV":10000}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), context)
for k in [.1, .4, 2.]:
    result = context["jnd_from_fit"](1, 0, 2.3, k)
    assert result["jnd_status"] == "ok" and math.isclose(result["jnd"], math.log(3)/k, rel_tol=0, abs_tol=2e-14)
assert math.isnan(context["jnd_from_fit"](.55, .35, .5, .4)["jnd"])
for parameters in [(1,0,0,0), (.3,.6,0,.4), (.5,.5,0,.4)]:
    assert context["jnd_from_fit"](*parameters)["jnd_status"] == "invalid_fit"
context["curve_fit"] = Mock(side_effect=RuntimeError("independent synthetic optimizer failure"))
failed = context["fit_condition"](np.array(SIGNED_ILDS), np.linspace(.1,.9,10))
assert failed["fit_status"] == "optimizer_failed" and math.isnan(failed["jnd"])


# %% Memberships, counts, and consistency with the existing RT-window chronometrics
members = pd.read_csv(OUTPUT / "group_membership.csv")
assert len(members) == 106
for r in members.itertuples(index=False):
    assert r.window_trials == member_counts[(r.view, r.group, r.filename, r.family, r.genotype, r.animal)]
for view, rows in members.groupby("view"):
    assert set(rows.animal) == set(identity) and rows.window_trials.sum() == 446949
old = pd.read_csv(REPO / "asd_analysis/session_type_1_genotype_chronometrics/group_membership.csv")
pd.testing.assert_frame_equal(members, old, check_dtype=False)
old_conditions = pd.read_csv(REPO / "asd_analysis/session_type_1_genotype_chronometrics/animal_condition_means.csv")
keys = ["filename", "family", "animal", "genotype", "plot_ABL", "abs_ILD"]
pd.testing.assert_frame_equal(animal_audit[keys+["window_trials"]], old_conditions[keys+["window_trials"]], check_dtype=False)
groups = pd.read_csv(OUTPUT / "group_counts.csv")
assert len(groups) == 18
for r in groups.itertuples(index=False):
    m = members.loc[members.view.eq(r.view) & members.group.eq(r.group) & members.genotype.eq(r.genotype)]
    assert r.pool_animals == len(m) and r.window_trials == m.window_trials.sum()
print("Raw counts, 795 accuracies, 213 refits, 416 numerical crossings, 324 pooled points and 287 SEMs passed", flush=True)


# %% Inspect actual saved PDF centers, bars, categorical positions, and square panels
layout = pd.read_csv(OUTPUT / "panel_layout.csv")
plotted = pd.read_csv(OUTPUT / "plotted_points.csv")
assert len(layout) == 36 and len(plotted) == 324
pdf_reports = []
for fig in manifest["figures"]:
    metric, view = fig["metric"], fig["view"]
    boxes = layout.loc[layout.metric.eq(metric) & layout.view.eq(view)]
    drawn = plotted.loc[plotted.metric.eq(metric) & plotted.view.eq(view)]
    assert len(boxes) == fig["rows"]*3 and len(drawn) == fig["points"]
    assert boxes.group.drop_duplicates().tolist() == (COHORTS if view == "cohort" else FAMILIES)
    assert boxes.plot_ABL.tolist() == ABLS*fig["rows"]
    assert boxes.width_in.eq(2.45).all() and boxes.height_in.eq(2.45).all()
    np.testing.assert_allclose(boxes[["ylim_low", "ylim_high"]], [manifest["ylims"][metric]]*len(boxes), atol=1e-14, rtol=0)
    with fitz.open(REPO / fig["pdf"]) as document:
        assert len(document) == 1
        page = document[0]
        assert math.isclose(page.rect.width, fig["width_in"]*72, abs_tol=.001) and math.isclose(page.rect.height, fig["height_in"]*72, abs_tol=.001)
        drawings = page.get_drawings()
        for genotype, color in COLORS.items():
            rgb = tuple(int(color[i:i+2],16)/255 for i in [1,3,5])
            circles = [d for d in drawings if d["fill"] is not None and np.allclose(d["fill"], rgb, atol=1e-6) and any(item[0] == "c" for item in d["items"])]
            p = drawn.loc[drawn.genotype.eq(genotype)]
            assert len(circles) == len(p)+1
            centers = np.array([[(d["rect"].x0+d["rect"].x1)/2, (d["rect"].y0+d["rect"].y1)/2] for d in circles])
            bars = []
            for d in drawings:
                if d["fill"] is None and d["color"] is not None and np.allclose(d["color"], rgb, atol=1e-6):
                    for item in d["items"]:
                        if item[0] == "l":
                            assert abs(item[1].x-item[2].x) < 1e-4 or (abs(item[1].y-item[2].y) < 1e-4 and abs(item[1].x-item[2].x) <= 4.01)
                            if abs(item[1].x-item[2].x) < 1e-4:
                                bars.append([item[1].x, min(item[1].y,item[2].y), max(item[1].y,item[2].y)])
            bars = np.array(bars)
            assert len(bars) == p["sem"].notna().sum()
            for r in p.itertuples(index=False):
                e = expected_points[(metric, view, r.group, genotype, r.plot_ABL, r.abs_ILD if metric == "accuracy" else None)]
                mean, sem, pool_n, sem_n, position = e
                assert r.color == color and (r.pool_animals, r.sem_animals) == (pool_n, sem_n)
                box = boxes.loc[boxes.group.eq(r.group) & boxes.plot_ABL.eq(r.plot_ABL)].iloc[0]
                x = (box.x_in+(position-box.xlim_low)/(box.xlim_high-box.xlim_low)*box.width_in)*72
                scale = box.height_in/(box.ylim_high-box.ylim_low)
                y = (fig["height_in"]-box.y_in-(mean-box.ylim_low)*scale)*72
                np.testing.assert_allclose([r.pdf_x_pt,r.pdf_y_pt], [x,y], rtol=0, atol=2e-8)
                assert (np.max(abs(centers-[x,y]),axis=1) < .02).sum() == 1
                if math.isfinite(sem):
                    top = (fig["height_in"]-box.y_in-(mean+sem-box.ylim_low)*scale)*72
                    bottom = (fig["height_in"]-box.y_in-(mean-sem-box.ylim_low)*scale)*72
                    np.testing.assert_allclose([r.pdf_bar_top_pt,r.pdf_bar_bottom_pt],[top,bottom],atol=2e-8,rtol=0)
                    assert (np.max(abs(bars-[x,top,bottom]),axis=1) < .02).sum() == 1
                    assert box.ylim_low <= mean-sem <= mean+sem <= box.ylim_high
                else:
                    assert math.isnan(r.pdf_bar_top_pt) and math.isnan(r.pdf_bar_bottom_pt)
                    panel_top, panel_bottom = (fig["height_in"]-box.y_in-box.height_in)*72, (fig["height_in"]-box.y_in)*72
                    assert not len(bars) or not np.any((abs(bars[:,0]-x) < .02) & (bars[:,1] >= panel_top) & (bars[:,2] <= panel_bottom))
        for block in page.get_text("blocks"):
            assert 0 <= block[0] <= block[2] <= page.rect.width and 0 <= block[1] <= block[3] <= page.rect.height
        lines = [line for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"]]
        labels = [line["bbox"] for line in lines if "".join(s["text"] for s in line["spans"]) in {"Genotype", "|ILD| (dB)"}]
        footer = [line["bbox"] for line in lines if "".join(s["text"] for s in line["spans"]).startswith("Dots:")]
        assert labels and len(footer) == 1 and max(box[3] for box in labels)+4 < footer[0][1]
        if metric == "jnd":
            text = page.get_text()
            for r in drawn.itertuples(index=False):
                assert f"pool {r.pool_animals}" in text and f"SEM {r.sem_animals}" in text
    with Image.open(REPO / fig["png"]) as image:
        assert abs(image.width-fig["width_in"]*200) <= 1 and abs(image.height-fig["height_in"]*200) <= 1
        image.verify()
    pdf_reports.append({"metric":metric, "view":view, "pages":1, "dots_checked":fig["points"], "bars_checked":fig["available_SEMs"]})
for relative, digest in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == digest, relative
report = {"status":"passed", "source_script":str(Path(__file__).relative_to(REPO)), "selected_trials_per_view":446949,
    "animals_per_view":53, "animal_accuracies_checked":795, "pooled_accuracies_checked":270, "accuracy_SEMs_checked":240,
    "accuracy_SEMs_unavailable":30, "refits_checked":213, "defined_animal_JNDs":154, "undefined_animal_JNDs":5,
    "pooled_JNDs_checked":54, "numerical_crossings_checked":roots_checked, "JND_SEMs_checked":47, "JND_SEMs_unavailable":7,
    "PDF_dots_checked":324, "PDF_bars_checked":287, "PDF_missing_bars_checked":37, "figures":pdf_reports,
    "protected_files_checked":len(manifest["protected_file_sha256"]),
    "synthetic_checks":["Zero-lapse JND=ln(3)/k", "Undefined thresholds", "Non-increasing/zero-k JND", "Optimizer failure"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
