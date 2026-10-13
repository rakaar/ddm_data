# %% Imports and paths
"""Independently count CSV choices and check saved sigmoid/JND figure artifacts.

Run after plot_session_type_1_slope_jnd.py, using .venv/bin/python.
The standard-library CSV parser reconstructs conditions and memberships;
numerical root finding checks crossings without the plotting script's inverse.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_slope_jnd"
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
EXPECTED_GROUPS = {"animal": 53, "filename_genotype": 12, "family_genotype": 6, "genotype": 3}
EXPECTED_FITS = {"animal": 159, "filename_genotype": 36, "family_genotype": 18, "genotype": 9}
COLORS = {20: (31/255, 119/255, 180/255), 40: (1, 127/255, 14/255), 60: (44/255, 160/255, 44/255)}
manifest = json.loads((OUTPUT / "slope_jnd_validation.json").read_text())
assert manifest["status"] == "passed"


# %% Reconstruct raw counts independently, with no RT selection
counts = defaultdict(lambda: [0, 0, 0])  # choices, right, correct
condition_animals = defaultdict(set)
members = defaultdict(lambda: [0, 0])
group_animals = defaultdict(set)
all_animals, chosen_animals = set(), set()
prepared_rows = selected_rows = right_rows = nonfinite_rt = 0
with INPUT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        prepared_rows += 1
        all_animals.add(row["animal"])
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"] == "False"
        assert float(row["is_short_sound"]) == 0 and float(row["short_duration"]) == 0
        success, abl, ild = float(row["success"]), float(row["ABL"]), float(row["ILD"])
        abl = 60 if abl == 59 else abl
        if success not in [1, -1] or abl not in ABLS or ild not in ILDS:
            continue
        response = float(row["response_poke"])
        assert response in [1, -1]
        right, correct = int(response == 1), int(success == 1)
        selected_rows += 1
        right_rows += right
        chosen_animals.add(row["animal"])
        try:
            nonfinite_rt += int(not np.isfinite(float(row["timed_rt"])))
        except ValueError:
            nonfinite_rt += 1
        group_names = {"animal": row["filename"] + "/" + row["animal"],
                       "filename_genotype": row["filename"],
                       "family_genotype": row["filename"].split("_")[0],
                       "genotype": "all_cohorts"}
        for view, group in group_names.items():
            key = (view, group, row["genotype"], int(abl), int(ild))
            counts[key][0] += 1
            counts[key][1] += right
            counts[key][2] += correct
            condition_animals[key].add(row["animal"])
            member_key = (view, group, row["genotype"], row["filename"], row["animal"])
            members[member_key][0] += 1
            members[member_key][1] += right
            group_animals[(view, group, row["genotype"])].add(row["animal"])
assert prepared_rows == 642370 and all_animals == chosen_animals and len(all_animals) == 53
assert (selected_rows, right_rows, selected_rows - right_rows) == (471416, 241539, 229877)
assert nonfinite_rt == 2188 and {"ASD0026", "ASD0045"} <= chosen_animals
assert len(counts) == 2220 and len(members) == 212


# %% Check every condition, membership, and group against the saved audits
points = pd.read_csv(OUTPUT / "slope_jnd_condition_counts.csv")
assert len(points) == len(counts)
assert not points.duplicated(["view", "group", "genotype", "plot_ABL", "ILD"]).any()
for row in points.itertuples():
    key = (row.view, row.group, row.genotype, int(row.plot_ABL), int(row.ILD))
    n, right, correct = counts[key]
    assert [row.trials, row.right_trials, row.left_trials, row.correct_trials, row.error_trials, row.animals] == [
        n, right, n-right, correct, n-correct, len(condition_animals[key])]
    assert np.isclose(row.p_right, right/n, rtol=0, atol=2e-16)
membership = pd.read_csv(OUTPUT / "slope_jnd_group_membership.csv")
assert len(membership) == 212
for row in membership.itertuples():
    key = (row.view, row.group, row.genotype, row.filename, row.animal)
    assert [row.trials, row.right_trials] == members[key]
for view in EXPECTED_GROUPS:
    assert set(membership.loc[membership.view.eq(view), "animal"]) == chosen_animals
    assert membership.loc[membership.view.eq(view), "trials"].sum() == 471416
groups = pd.read_csv(OUTPUT / "slope_jnd_group_counts.csv")
assert groups.groupby("view").size().to_dict() == EXPECTED_GROUPS
for row in groups.itertuples():
    assert row.animals == len(group_animals[(row.view, row.group, row.genotype)])
    subset = membership.loc[membership.view.eq(row.view) & membership.group.eq(row.group)
                            & membership.genotype.eq(row.genotype)]
    assert subset[["trials", "right_trials"]].sum().tolist() == [row.trials, row.right_trials]


# %% Check fits, unweighted residuals, thresholds, and numerical crossings
fits = pd.read_csv(OUTPUT / "slope_jnd_fits.csv")
assert len(fits) == 222 and fits.groupby("view").size().to_dict() == EXPECTED_FITS
assert not fits.duplicated(["view", "group", "genotype", "plot_ABL"]).any()
crossings_checked = jnds_checked = 0
for row in fits.itertuples():
    condition = points.loc[points.view.eq(row.view) & points.group.eq(row.group)
                           & points.genotype.eq(row.genotype) & points.plot_ABL.eq(row.plot_ABL)].sort_values("ILD")
    assert condition.ILD.tolist() == ILDS and row.ILD_points == 10
    assert condition[["trials", "right_trials", "left_trials"]].sum().tolist() == [row.trials, row.right_trials, row.left_trials]
    assert row.animals == len(group_animals[(row.view, row.group, row.genotype)])
    np.testing.assert_allclose(json.loads(row.initial_parameters),
                               [condition.p_right.max(), condition.p_right.min(), 0, .1], rtol=0, atol=2e-16)
    if row.fit_status != "ok":
        assert not np.isfinite(row.slope_k) and not np.isfinite(row.jnd)
        continue
    assert np.isfinite([row.upper, row.lower, row.x0, row.k]).all()
    assert 0 <= row.lower < row.upper <= 1 and row.k > 0 and row.slope_k == row.k
    fitted = row.lower + (row.upper-row.lower) * expit(row.k * (condition.ILD.to_numpy()-row.x0))
    np.testing.assert_allclose(condition.fitted_p_right, fitted, rtol=0, atol=2e-14)
    residual = fitted-condition.p_right.to_numpy()
    np.testing.assert_allclose(condition.residual, residual, rtol=0, atol=2e-14)
    assert np.isclose(row.unweighted_sse, residual @ residual, rtol=0, atol=2e-14)
    assert np.isclose(row.rmse, np.sqrt(np.mean(residual**2)), rtol=0, atol=2e-14)
    assert np.isclose(row.y_high, .75*row.upper, rtol=0, atol=2e-16)
    assert np.isclose(row.y_low, .25*(1-row.lower), rtol=0, atol=2e-16)
    crossing_values = {}
    for side in ["high", "low"]:
        target = getattr(row, f"y_{side}")
        reported = getattr(row, f"ild_{side}")
        if row.lower < target < row.upper:
            z = brentq(lambda z: row.lower+(row.upper-row.lower)*expit(z)-target, -100, 100, xtol=1e-13)
            crossing_values[side] = row.x0+z/row.k
            assert np.isclose(reported, crossing_values[side], rtol=0, atol=5e-10)
            assert np.isclose(row.lower+(row.upper-row.lower)*expit(row.k*(reported-row.x0)), target, rtol=0, atol=2e-15)
            crossings_checked += 1
        else:
            assert not np.isfinite(reported)
    if row.lower < row.y_low < row.y_high < row.upper:
        independent_jnd = (crossing_values["high"]-crossing_values["low"])/2
        assert row.jnd_status == "ok" and independent_jnd > 0
        assert np.isclose(row.jnd, independent_jnd, rtol=0, atol=5e-10)
        jnds_checked += 1
    else:
        assert row.jnd_status == "thresholds_outside_asymptotes_or_reversed" and not np.isfinite(row.jnd)
        assert np.isfinite(row.slope_k)  # JND alone is unavailable.
for k in [.1, .4, 2]:
    z_high = brentq(lambda z: expit(z)-.75, -100, 100)
    z_low = brentq(lambda z: expit(z)-.25, -100, 100)
    assert np.isclose((z_high-z_low)/(2*k), np.log(3)/k, rtol=0, atol=1e-12)
status = fits.groupby(["view", "fit_status", "jnd_status"]).size().rename("count").reset_index()
pd.testing.assert_frame_equal(status, pd.read_csv(OUTPUT / "slope_jnd_fit_status_counts.csv"))
assert fits.fit_status.eq("ok").sum() == 222 and jnds_checked == 218 and crossings_checked == 440


# %% Saved PDF colors/dot counts, PNG sizes, full valid ranges, and preserved inputs
artifact_checks = []
expected_pixels = {"animal": (4680, 1440), "filename_genotype": (4320, 1530),
                   "family_genotype": (3060, 1440), "genotype": (2160, 1260)}
for page in manifest["pages"]:
    metric, view = page["metric"], page["view"]
    subset = fits.loc[fits.view.eq(view)]
    available = subset.loc[np.isfinite(subset[metric])]
    assert page["available_estimates"] == len(available)
    assert page["unavailable_estimates"] == len(subset)-len(available)
    assert page["ylim"][0] == 0 and page["ylim"][1] > available[metric].max()
    assert page["panels"] == 1
    if view != "animal":
        assert page["label_rotation"] == 45
        assert len(page["categorical_groups"]) == EXPECTED_GROUPS[view]
        assert np.allclose(np.diff([item["x"] for item in page["categorical_groups"]]), page["category_spacing"])
        assert page["category_spacing"] > 2 * max(abs(offset) for offset in manifest["animal_dot_offsets"].values())
        for category in page["categorical_groups"]:
            assert category["animals"] == len(group_animals[(view, category["group"], category["genotype"])])
    with Image.open(REPO / page["png"]) as png:
        assert png.size == expected_pixels[view]
        png.verify()
    with fitz.open(REPO / page["pdf"]) as pdf:
        assert pdf.page_count == 1
        text = pdf[0].get_text()
        assert ("Slope = k" if metric == "slope_k" else "Figure 1 JND") in text
        assert text.count("k (dB" if metric == "slope_k" else "JND (dB)") == 1
        drawings = pdf[0].get_drawings()
        color_counts = {}
        for abl, rgb in COLORS.items():
            paths = [drawing for drawing in drawings if drawing["fill"] is not None
                     and np.allclose(drawing["fill"], rgb, rtol=0, atol=2e-6)]
            expected_dots = int(available.plot_ABL.eq(abl).sum())
            assert len(paths) == expected_dots+1  # Exactly one extra colored marker in the legend.
            color_counts[str(abl)] = expected_dots
        if view == "animal":
            assert all(animal in text for animal in chosen_animals)
    artifact_checks.append({"view": view, "metric": metric, "pdf_pages": 1,
                            "axes": 1, "categorical_positions": EXPECTED_GROUPS[view],
                            "label_rotation": page["label_rotation"],
                            "png_pixels": expected_pixels[view], "saved_colored_dots": color_counts})
for name, digest in manifest["protected_file_sha256_unchanged"].items():
    with (REPO / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
assert hashlib.sha256((REPO / manifest["source_script"]).read_bytes()).hexdigest() == manifest["source_script_sha256"]
assert manifest["all_pooled_conditions_match_existing_psychometric_audits"]

report = {
    "status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_sha256": manifest["input_sha256"], "prepared_rows": prepared_rows,
    "choices_per_view": selected_rows, "right_choices_per_view": right_rows,
    "left_choices_per_view": selected_rows-right_rows, "animals_per_view": len(chosen_animals),
    "included_nonfinite_timed_rt_trials": nonfinite_rt, "groups_per_view": EXPECTED_GROUPS,
    "fits_per_view": EXPECTED_FITS, "condition_points_checked": len(points),
    "membership_records_checked": len(membership), "attempted_fits_checked": len(fits),
    "valid_k_estimates": 222, "valid_jnd_estimates": jnds_checked,
    "numerical_crossings_checked": crossings_checked, "unavailable_jnd_records": fits.loc[fits.jnd_status.ne("ok"),
        ["view", "group", "genotype", "plot_ABL", "jnd_status"]].to_dict(orient="records"),
    "checks": ["independent CSV counts", "all memberships and group counts", "no RT restriction",
               "slope exactly k", "initial guesses and bounds", "unweighted residuals and SSE",
               "crossings independently reconstructed by brentq", "zero-lapse JND identity",
               "valid k retained for unavailable JND", "saved PDF color and marker counts",
               "one-page PDFs and PNG sizes", "full valid ranges", "protected CSVs and figures unchanged"],
    "artifact_checks": artifact_checks,
}
(OUTPUT / "slope_jnd_independent_validation.json").write_text(json.dumps(report, indent=2)+"\n")
print("Independent verification passed: 2,220 conditions, 222 fits, 440 crossings, and eight PNG/PDF pairs.")
