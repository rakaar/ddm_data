# %% Imports and parameters
"""Independently verify genotype overlays from raw CSV choice counts and saved PDFs.

Run after plot_session_type_1_genotype_overlay_psychometrics.py with .venv/bin/python.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO/"asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO/"asd_analysis/session_type_1_genotype_overlay_psychometrics"
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
METHODS = ["trial_pooled", "equal_abl_average"]
EXPECTED_GROUPS = {"filename_genotype": 12, "family_genotype": 6, "genotype": 3}
EXPECTED_PIXELS = {"filename_genotype": (2520, 1800), "family_genotype": (2880, 1080), "genotype": (2160, 1260)}
COLORS = {"wt": (31/255, 119/255, 180/255), "het": (1, 127/255, 14/255), "hom": (44/255, 160/255, 44/255)}
manifest = json.loads((OUTPUT/"overlay_validation.json").read_text())
assert manifest["status"] == "passed"


# %% Reconstruct ABL counts and memberships with the standard-library CSV parser
counts = defaultdict(lambda: [0, 0, 0])  # choices, right, correct
cell_animals = defaultdict(set)
members = defaultdict(lambda: [0, 0])
animals, selected_animals = set(), set()
prepared_rows = choices = right_choices = missing_rt = 0
with INPUT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        prepared_rows += 1
        animals.add(row["animal"])
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"] == "False"
        assert float(row["is_short_sound"]) == 0 and float(row["short_duration"]) == 0
        success, abl, ild = float(row["success"]), float(row["ABL"]), float(row["ILD"])
        abl = 60 if abl == 59 else abl
        if success not in [1, -1] or abl not in ABLS or ild not in ILDS:
            continue
        response = float(row["response_poke"])
        assert response in [1, -1]
        choices += 1
        right = int(response == 1)
        right_choices += right
        selected_animals.add(row["animal"])
        try:
            missing_rt += int(not np.isfinite(float(row["timed_rt"])))
        except ValueError:
            missing_rt += 1
        groups = {"filename_genotype": row["filename"], "family_genotype": row["filename"].split("_")[0],
                  "genotype": "all_cohorts"}
        for view, group in groups.items():
            key = (view, group, row["genotype"], int(abl), int(ild))
            counts[key][0] += 1
            counts[key][1] += right
            counts[key][2] += int(success == 1)
            cell_animals[key].add(row["animal"])
            member_key = (view, group, row["genotype"], row["filename"], row["animal"])
            members[member_key][0] += 1
            members[member_key][1] += right
assert prepared_rows == 642370 and animals == selected_animals and len(animals) == 53
assert {"ASD0026", "ASD0045"} <= animals
assert (choices, right_choices, choices-right_choices, missing_rt) == (471416, 241539, 229877, 2188)
assert len(counts) == 630 and len(members) == 159


# %% Verify every ABL count, membership, and group count
abl_counts = pd.read_csv(OUTPUT/"overlay_abl_counts.csv")
assert len(abl_counts) == 630
assert not abl_counts.duplicated(["view", "group", "genotype", "plot_ABL", "ILD"]).any()
for row in abl_counts.itertuples():
    key = (row.view, row.group, row.genotype, int(row.plot_ABL), int(row.ILD))
    n, right, correct = counts[key]
    assert [row.trials, row.right_trials, row.left_trials, row.correct_trials, row.error_trials, row.animals] == [
        n, right, n-right, correct, n-correct, len(cell_animals[key])]
    assert np.isclose(row.p_right, right/n, rtol=0, atol=2e-16)
membership = pd.read_csv(OUTPUT/"overlay_group_membership.csv")
assert len(membership) == 159
assert not membership.duplicated(["view", "group", "genotype", "filename", "animal"]).any()
for row in membership.itertuples():
    assert [row.trials, row.right_trials] == members[(row.view, row.group, row.genotype, row.filename, row.animal)]
groups = pd.read_csv(OUTPUT/"overlay_group_counts.csv")
assert groups.groupby("view").size().to_dict() == EXPECTED_GROUPS
for view in EXPECTED_GROUPS:
    assert set(membership.loc[membership.view.eq(view), "animal"]) == animals
    assert groups.loc[groups.view.eq(view), ["trials", "right_trials", "left_trials", "animals"]].sum().tolist() == [471416, 241539, 229877, 53]
for row in groups.itertuples():
    subset = membership.loc[membership.view.eq(row.view) & membership.group.eq(row.group) & membership.genotype.eq(row.genotype)]
    assert [row.trials, row.right_trials, row.animals] == [int(subset.trials.sum()), int(subset.right_trials.sum()), subset.animal.nunique()]


# %% Independently derive both probabilities and their exact ABL weights
points = pd.read_csv(OUTPUT/"overlay_combined_points.csv")
assert len(points) == 420 and points.groupby("method").size().eq(210).all()
assert not points.duplicated(["method", "view", "group", "genotype", "ILD"]).any()
for row in points.itertuples():
    sources = [counts[(row.view, row.group, row.genotype, abl, int(row.ILD))] for abl in ABLS]
    n, right, correct = (sum(source[index] for source in sources) for index in [0, 1, 2])
    source_probabilities = [source[1]/source[0] for source in sources]
    condition_animals = set().union(*(cell_animals[(row.view, row.group, row.genotype, abl, int(row.ILD))] for abl in ABLS))
    assert [row.trials, row.right_trials, row.left_trials, row.correct_trials, row.error_trials, row.animals] == [
        n, right, n-right, correct, n-correct, len(condition_animals)]
    probability = right/n if row.method == "trial_pooled" else sum(source_probabilities)/3
    assert np.isclose(row.p_right, probability, rtol=0, atol=2e-15)
    for index, abl in enumerate(ABLS):
        expected_weight = sources[index][0]/n if row.method == "trial_pooled" else 1/3
        assert getattr(row, f"ABL_{abl}_trials") == sources[index][0]
        assert getattr(row, f"ABL_{abl}_right_trials") == sources[index][1]
        assert np.isclose(getattr(row, f"ABL_{abl}_p_right"), source_probabilities[index], rtol=0, atol=2e-16)
        assert np.isclose(getattr(row, f"ABL_{abl}_weight"), expected_weight, rtol=0, atol=2e-16)
for (method, view), subset in points.groupby(["method", "view"]):
    assert subset[["trials", "right_trials", "left_trials"]].sum().tolist() == [471416, 241539, 229877]
comparison = points.pivot(index=["view", "group", "genotype", "ILD"], columns="method", values="p_right")
maximum_method_difference = float((comparison.trial_pooled-comparison.equal_abl_average).abs().max())
assert maximum_method_difference > .01  # Actual unequal ABL counts must produce different methods.


# %% Verify fitted probabilities, four-parameter bounds, and unweighted residuals
fits = pd.read_csv(OUTPUT/"overlay_fits.csv")
assert len(fits) == 42 and fits.groupby("method").size().to_dict() == {method: 21 for method in METHODS}
assert not fits.duplicated(["method", "view", "group", "genotype"]).any()
for row in fits.itertuples():
    condition = points.loc[points.method.eq(row.method) & points.view.eq(row.view) & points.group.eq(row.group)
                           & points.genotype.eq(row.genotype)].sort_values("ILD")
    assert condition.ILD.tolist() == ILDS and row.ILD_points == 10
    assert condition[["trials", "right_trials", "left_trials"]].sum().tolist() == [row.trials, row.right_trials, row.left_trials]
    np.testing.assert_allclose(json.loads(row.initial_parameters),
                               [condition.p_right.max(), condition.p_right.min(), 0, .1], rtol=0, atol=2e-15)
    if row.fit_status != "ok":
        assert condition.fitted_p_right.isna().all()
        continue
    assert np.isfinite([row.upper, row.lower, row.x0, row.k]).all()
    assert 0 <= row.lower < row.upper <= 1 and row.k > 0
    assert row.optimizer_code in [1, 2, 3, 4]
    predicted = row.lower+(row.upper-row.lower)*expit(row.k*(condition.ILD.to_numpy()-row.x0))
    residual = predicted-condition.p_right.to_numpy()
    np.testing.assert_allclose(condition.fitted_p_right, predicted, rtol=0, atol=2e-14)
    np.testing.assert_allclose(condition.residual, residual, rtol=0, atol=2e-14)
    assert np.isclose(row.unweighted_sse, residual @ residual, rtol=0, atol=2e-14)
    assert np.isclose(row.rmse, np.sqrt(np.mean(residual**2)), rtol=0, atol=2e-14)
status = fits.groupby(["method", "view", "fit_status"]).size().rename("count").reset_index()
pd.testing.assert_frame_equal(status, pd.read_csv(OUTPUT/"overlay_fit_status_counts.csv"))


# %% Check saved PDF colors, scatter-marker and fitted-line counts, PNG dimensions
artifact_checks = []
for page in manifest["pages"]:
    view, method, panels = page["view"], page["method"], page["panels"]
    assert panels == EXPECTED_GROUPS[view]/3
    with Image.open(REPO/page["png"]) as png:
        assert png.size == EXPECTED_PIXELS[view]
        png.verify()
    with fitz.open(REPO/page["pdf"]) as pdf:
        assert pdf.page_count == 1
        text = pdf[0].get_text()
        assert sum(line.strip() == "P(right)" for line in text.splitlines()) == panels
        assert text.count("eligible choices") == panels
        assert manifest["methods"][method]["title"] in text
        drawings = pdf[0].get_drawings()
        marker_counts, curve_counts = {}, {}
        for genotype, rgb in COLORS.items():
            colored_markers = [drawing for drawing in drawings if drawing["fill"] is not None
                and np.allclose(drawing["fill"], rgb, atol=2e-6, rtol=0)]
            expected_markers = 10*panels+panels+1  # Ten observed points per panel plus local/global legend markers.
            assert len(colored_markers) == expected_markers, (method, view, genotype, len(colored_markers))
            curves = [drawing for drawing in drawings if drawing["fill"] is None and drawing["color"] is not None
                and np.allclose(drawing["color"], rgb, atol=2e-6, rtol=0) and np.isclose(drawing["width"], 1.8, atol=2e-6, rtol=0)]
            expected_curves = int((fits.method.eq(method) & fits.view.eq(view) & fits.genotype.eq(genotype) & fits.fit_status.eq("ok")).sum())
            assert len(curves) == expected_curves, (method, view, genotype, len(curves), expected_curves)
            marker_counts[genotype] = 10*panels
            curve_counts[genotype] = expected_curves
    assert page["scatter_points"] == 30*panels and page["xlim"] == [-18, 18] and page["ylim"] == [0, 1]
    artifact_checks.append({"method": method, "view": view, "pdf_pages": 1, "panels": panels,
                            "png_pixels": EXPECTED_PIXELS[view], "observed_marker_counts": marker_counts,
                            "saved_fit_curve_counts": curve_counts})
for name, digest in manifest["protected_file_sha256_unchanged"].items():
    with (REPO/name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
assert hashlib.sha256((REPO/manifest["source_script"]).read_bytes()).hexdigest() == manifest["source_script_sha256"]
report = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_sha256": manifest["input_sha256"], "prepared_rows": prepared_rows,
    "animals_per_view_and_method": len(animals), "choices_per_view_and_method": choices,
    "right_choices": right_choices, "left_choices": choices-right_choices, "included_nonfinite_RT_choices": missing_rt,
    "abl_condition_points_checked": len(abl_counts), "combined_points_checked": len(points),
    "membership_records_checked": len(membership), "fits_checked": len(fits),
    "valid_fits": int(fits.fit_status.eq("ok").sum()), "maximum_pooled_vs_equal_ABL_probability_difference": maximum_method_difference,
    "checks": ["independent raw CSV counts", "memberships and group totals", "pooled trial fractions",
               "equal means of measured ABL probabilities", "all ABL weights", "ten ILDs per fit",
               "initial guesses and bounds", "unweighted residuals", "saved PDF marker/fit-line colors and counts",
               "single-page PDFs and PNG sizes", "unchanged inputs and all previous analyses"],
    "artifact_checks": artifact_checks,
}
(OUTPUT/"overlay_independent_validation.json").write_text(json.dumps(report, indent=2)+"\n")
print("Independent verification passed: 630 ABL cells, 420 combined points, 42 fits, and six PNG/PDF pairs.")
