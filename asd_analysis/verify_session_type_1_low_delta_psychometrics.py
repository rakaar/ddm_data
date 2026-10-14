# %% Imports and parameters
"""Independently check ranking arithmetic, reused probabilities/fits, and exports.

Run with .venv/bin/python after plot_session_type_1_low_delta_psychometrics.py.
Use stdlib CSV parsing and combinations for ranking arithmetic; do not refit.
"""
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_low_delta_psychometrics"
manifest = json.loads((OUTPUT / "validation.json").read_text())
REFERENCE = REPO / manifest["reference_folder"]
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
assert manifest["status"] == "passed" and manifest["refitted"] is False
assert manifest["rt_restriction"] is None and manifest["repeat_filter_reapplied"] is False


# %% Independently reconstruct all 53 animals and both rankings from saved fits
source_fits, source_points = {}, {}
with (REFERENCE / "slope_jnd_fits.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row["view"] != "animal":
            continue
        key = (row["group"], row["genotype"], int(row["plot_ABL"]))
        source_fits[key] = row
with (REFERENCE / "slope_jnd_condition_counts.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row["view"] != "animal":
            continue
        key = (row["group"], row["genotype"], int(float(row["plot_ABL"])), int(float(row["ILD"])))
        assert key not in source_points
        source_points[key] = row
assert len(source_fits) == 159 and len(source_points) == 1590
audit = pd.read_csv(OUTPUT / "animal_eligibility_metrics.csv", keep_default_na=False)
assert len(audit) == 53 and audit.animal.nunique() == 53
eligible, independent = set(), {}
for animal in audit.itertuples(index=False):
    group = f"{animal.filename}/{animal.animal}"
    assert animal.group == group
    estimates = [source_fits[(group, animal.genotype, abl)] for abl in ABLS]
    jnd_values = [float(f["jnd"]) if f["jnd"] else math.nan for f in estimates]
    k_values = [float(f["k"]) for f in estimates]
    is_eligible = all(math.isfinite(v) and v > 0 for v in jnd_values + k_values)
    assert bool(animal.common_eligible) == is_eligible
    assert bool(animal.jnd_eligible) == all(math.isfinite(v) for v in jnd_values)
    assert bool(animal.slope_eligible) is True
    if is_eligible:
        eligible.add(animal.animal)
    for metric, values, units in [("jnd", jnd_values, "dB"), ("k", k_values, "per_dB")]:
        differences = [abs(a-b) for a,b in combinations(values, 2)]
        score = math.fsum(differences) / 3
        actual = getattr(animal, f"delta_{metric}_{units}")
        if math.isfinite(score):
            assert math.isclose(float(actual), score, rel_tol=0, abs_tol=2e-15)
            assert math.isclose(score, 2/3 * (max(values)-min(values)), rel_tol=0, abs_tol=2e-15)
            independent[(animal.animal, metric)] = score
        else:
            assert actual == ""  # Undefined JND is not filled or averaged over fewer ABLs.
        for abl, value in zip(ABLS, values):
            actual = getattr(animal, f"{metric}_ABL_{abl}")
            assert actual == "" if not math.isfinite(value) else math.isclose(float(actual), value, rel_tol=0, abs_tol=2e-15)
        for a,b in [(20,40),(40,60),(20,60)]:
            expected = abs(values[ABLS.index(a)]-values[ABLS.index(b)])
            actual = getattr(animal, f"abs_{metric}_{a}_minus_{b}_{units}")
            assert actual == "" if not math.isfinite(expected) else math.isclose(float(actual), expected, rel_tol=0, abs_tol=2e-15)
    conditions = [source_points[(group,animal.genotype,abl,ild)] for abl in ABLS for ild in ILDS]
    assert animal.choice_trials == sum(int(p["trials"]) for p in conditions)
    assert animal.right_trials == sum(int(p["right_trials"]) for p in conditions)
    assert animal.left_trials == sum(int(p["left_trials"]) for p in conditions)
assert len(eligible) == 50
assert set(audit.animal)-eligible == {"ASD0035", "ASD0044", "ASD0048"}
assert audit.loc[audit.animal.isin(eligible), "choice_trials"].sum() == 444718
for metric in ["jnd", "k"]:
    ranking = pd.read_csv(OUTPUT / f"{metric}_ranking.csv")
    expected_order = sorted(eligible, key=lambda animal: (independent[(animal,metric)], animal))
    assert ranking.animal.to_list() == expected_order
    assert ranking["rank"].to_list() == list(range(1,51))
    assert expected_order[:5] == manifest["top_animals"][metric]
    assert set(ranking.animal) == eligible


# %% Check all 300 measured points, 30 reused fits, and their full smooth curves
points = pd.read_csv(OUTPUT / "selected_psychometric_points.csv")
fits = pd.read_csv(OUTPUT / "selected_psychometric_fits.csv")
curves = pd.read_csv(OUTPUT / "selected_sigmoid_curves.csv")
counts = pd.read_csv(OUTPUT / "selected_panel_counts.csv")
assert len(points) == 300 and len(fits) == len(counts) == 30
assert points.groupby(["panel_row", "panel_column"]).size().eq(30).all()
assert points.animal.nunique() == 9
for point in points.itertuples(index=False):
    source = source_points[(point.group,point.genotype,int(point.plot_ABL),int(point.ILD))]
    for field in ["trials", "right_trials", "left_trials", "correct_trials", "error_trials"]:
        assert getattr(point,field) == int(source[field])
    assert math.isclose(point.p_right,point.right_trials/point.trials,rel_tol=0,abs_tol=2e-16)
    assert math.isclose(point.p_right,float(source["p_right"]),rel_tol=0,abs_tol=2e-16)
    assert math.isclose(point.fitted_p_right,float(source["fitted_p_right"]),rel_tol=0,abs_tol=2e-16)
for fit in fits.itertuples(index=False):
    source = source_fits[(fit.group,fit.genotype,int(fit.plot_ABL))]
    for parameter in ["upper", "lower", "x0", "k", "slope_k", "jnd", "y_high", "y_low", "ild_high", "ild_low"]:
        assert math.isclose(getattr(fit,parameter),float(source[parameter]),rel_tol=0,abs_tol=2e-14)
    assert fit.fit_status == fit.jnd_status == "ok"
    smooth = curves.loc[curves.selection_metric.eq(fit.selection_metric) & curves.animal.eq(fit.animal)
                        & curves.plot_ABL.eq(fit.plot_ABL)]
    assert len(smooth) == 501 and smooth.ILD.min() == -18 and smooth.ILD.max() == 18
    np.testing.assert_allclose(smooth.fitted_p_right,
        fit.lower+(fit.upper-fit.lower)*expit(fit.k*(smooth.ILD-fit.x0)),rtol=0,atol=2e-14)
for count in counts.itertuples(index=False):
    subset=points.loc[points.selection_metric.eq(count.selection_metric) & points.animal.eq(count.animal)
                      & points.plot_ABL.eq(count.plot_ABL)]
    assert subset.ILD.to_list() == ILDS
    assert count.trials == subset.trials.sum() and count.right_trials == subset.right_trials.sum()
    assert count.left_trials == subset.left_trials.sum()
    assert count.animal == manifest["top_animals"][count.selection_metric][count.selection_rank-1]
assert counts.groupby("selection_metric").trials.sum().to_dict() == {"jnd":43490,"k":41474}


# %% Verify the saved PDF marker/curve counts, colors, layout, and animal identities
with Image.open(REPO / manifest["png"]) as png:
    assert png.size == (3600,1440)
    png.verify()
with fitz.open(REPO / manifest["pdf"]) as document:
    assert len(document) == 1
    page = document[0]
    assert np.isclose(page.rect.width,20*72) and np.isclose(page.rect.height,8*72)
    text = page.get_text()
    assert set(re.findall(r"\bASD\d{4}\b",text)) == set(points.animal)
    assert len(re.findall(r"Rank [1-5] \|", text)) == 10 and text.count("n=") == 30
    paths=page.get_drawings()
    artifact_counts={}
    for abl,color in manifest["colors"].items():
        rgb=np.array([int(color[i:i+2],16)/255 for i in [1,3,5]])
        markers=[p for p in paths if p["fill"] is not None and np.allclose(p["fill"],rgb,atol=2e-6,rtol=0)]
        # 100 measured points, ten local legend markers, and one page legend marker.
        assert len(markers) == 111,(abl,len(markers))
        lines=[p for p in paths if p["fill"] is None and p["color"] is not None
               and np.allclose(p["color"],rgb,atol=2e-6,rtol=0) and np.isclose(p["width"],1.7,atol=1e-5)]
        assert len(lines) == 10,(abl,len(lines))
        artifact_counts[abl]={"measured_points":100,"fitted_curves":10,"legend_markers":11}
assert len(manifest["axis_bounds"]) == 10
assert manifest["xlim"] == [-18,18] and manifest["ylim"] == [0,1]
for relative,digest in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == digest,relative
report={"status":"passed","source_script":str(Path(__file__).relative_to(REPO)),
    "animals_audited":53,"common_eligible_animals":50,"eligible_choices":444718,
    "top_animals":manifest["top_animals"],"selected_row_choices":{"jnd":43490,"k":41474},
    "rankings_checked":2,"panels":10,"measured_points_checked":300,"saved_fits_checked":30,
    "smooth_curve_points_checked":len(curves),"pdf_pages":1,"png_pixels":[3600,1440],
    "artifact_counts":artifact_counts,"protected_files_checked":len(manifest["protected_file_sha256"]),
    "checks":["stdlib CSV parsing of source fits/counts", "all pairwise differences and two-thirds-range identity",
        "missing JNDs stay unavailable; common eligibility for both rankings", "ascending scores with animal-ID tie break",
        "all 300 measured probabilities/counts match source", "all 30 fits and smooth curves match source parameters",
        "saved PDF markers/curves/colors and identities", "one 20-by-8-inch PDF and matching PNG", "protected hashes unchanged"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print("Independent verification passed: 50 eligible animals; 10 panels, 30 curves, 300 points; preserved inputs/previous analyses.")
