# %% Imports and parameters
"""Independent CSV/fit/PDF verification for measured-RT-restricted psychometrics."""
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
OUTPUT = REPO/"asd_analysis/session_type_1_genotype_overlay_psychometrics_rt_0_to_1"
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
EXPECTED_GROUPS = {"filename_genotype": 12, "family_genotype": 6, "genotype": 3}
EXPECTED_PIXELS = {"filename_genotype": (2520, 1800), "family_genotype": (2880, 1080), "genotype": (2160, 1260)}
COLORS = {"wt": (31/255, 119/255, 180/255), "het": (1, 127/255, 14/255), "hom": (44/255, 160/255, 44/255)}
manifest = json.loads((OUTPUT/"rt_window_validation.json").read_text())
assert manifest["status"] == "passed"


# %% Independently filter raw prepared rows and reconstruct all choice counts
counts = defaultdict(lambda: [0, 0, 0])  # trials, right, correct at signed ILD.
abl_counts = defaultdict(int)
cell_animals = defaultdict(set)
members = defaultdict(lambda: [0, 0])
all_animals, chosen_animals = set(), set()
prepared = base_choices = chosen = right_choices = nonfinite = above = below = 0
with INPUT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        prepared += 1
        all_animals.add(row["animal"])
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"] == "False" and float(row["is_short_sound"]) == 0
        assert float(row["max_rt"]) == 10
        success, abl, ild = float(row["success"]), float(row["ABL"]), float(row["ILD"])
        abl = 60 if abl == 59 else abl
        if success not in [1, -1] or abl not in ABLS or ild not in ILDS:
            continue
        base_choices += 1
        try:
            rt = float(row["timed_rt"])
        except ValueError:
            rt = float("nan")
        if not np.isfinite(rt):
            nonfinite += 1
            continue
        if rt < 0:
            below += 1
            continue
        if rt > 1:
            above += 1
            continue
        response = float(row["response_poke"])
        assert response in [1, -1]
        right = int(response == 1)
        chosen += 1
        right_choices += right
        chosen_animals.add(row["animal"])
        for view, group in {"filename_genotype": row["filename"], "family_genotype": row["filename"].split("_")[0],
                            "genotype": "all_cohorts"}.items():
            key = (view, group, row["genotype"], int(ild))
            counts[key][0] += 1
            counts[key][1] += right
            counts[key][2] += int(success == 1)
            cell_animals[key].add(row["animal"])
            abl_counts[(view, group, row["genotype"], int(abl), int(ild))] += 1
            member_key = (view, group, row["genotype"], row["filename"], row["animal"])
            members[member_key][0] += 1
            members[member_key][1] += right
assert (prepared, base_choices, chosen, right_choices, chosen-right_choices) == (642370, 471416, 466978, 239331, 227647)
assert (nonfinite, below, above) == (2188, 0, 2250)
assert chosen_animals == all_animals and len(chosen_animals) == 53
assert len(counts) == 210 and len(abl_counts) == 630 and len(members) == 159


# %% Verify probabilities, ABL trial totals, group counts, and memberships
points = pd.read_csv(OUTPUT/"rt_window_points.csv")
assert len(points) == 210 and not points.duplicated(["view", "group", "genotype", "ILD"]).any()
for row in points.itertuples():
    key = (row.view, row.group, row.genotype, int(row.ILD))
    n, right, correct = counts[key]
    assert [row.trials, row.right_trials, row.left_trials, row.correct_trials, row.error_trials, row.animals] == [
        n, right, n-right, correct, n-correct, len(cell_animals[key])]
    assert np.isclose(row.p_right, right/n, rtol=0, atol=2e-16)
abl_table = pd.read_csv(OUTPUT/"rt_window_abl_counts.csv")
assert len(abl_table) == 630
for row in abl_table.itertuples():
    assert row.trials == abl_counts[(row.view, row.group, row.genotype, int(row.plot_ABL), int(row.ILD))]
membership = pd.read_csv(OUTPUT/"rt_window_membership.csv")
assert len(membership) == 159
for row in membership.itertuples():
    assert [row.trials, row.right_trials] == members[(row.view, row.group, row.genotype, row.filename, row.animal)]
groups = pd.read_csv(OUTPUT/"rt_window_group_counts.csv")
assert groups.groupby("view").size().to_dict() == EXPECTED_GROUPS
for view in EXPECTED_GROUPS:
    assert groups.loc[groups.view.eq(view), ["trials", "right_trials", "left_trials", "animals"]].sum().tolist() == [466978, 239331, 227647, 53]
    assert set(membership.loc[membership.view.eq(view), "animal"]) == chosen_animals
cohort2 = membership.loc[membership.view.eq("filename_genotype") & membership.filename.eq("SHANK3_cohort2.csv")]
assert set(cohort2.animal) == {"ASD0072", "ASD0073", "ASD0074", "ASD0075", "ASD0076", "ASD0077"}


# %% Verify the ten unweighted points and saved fit predictions independently
fits = pd.read_csv(OUTPUT/"rt_window_fits.csv")
assert len(fits) == 21 and fits.groupby("view").size().to_dict() == EXPECTED_GROUPS
for row in fits.itertuples():
    condition = points.loc[points.view.eq(row.view) & points.group.eq(row.group) & points.genotype.eq(row.genotype)].sort_values("ILD")
    assert condition.ILD.tolist() == ILDS and row.ILD_points == 10
    assert condition.trials.sum() == row.trials
    np.testing.assert_allclose(json.loads(row.initial_parameters), [condition.p_right.max(), condition.p_right.min(), 0, .1], rtol=0, atol=2e-15)
    if row.fit_status != "ok":
        assert condition.fitted_p_right.isna().all()
        continue
    assert 0 <= row.lower < row.upper <= 1 and row.k > 0 and np.isfinite([row.upper, row.lower, row.x0, row.k]).all()
    predicted = row.lower+(row.upper-row.lower)*expit(row.k*(condition.ILD.to_numpy()-row.x0))
    residual = predicted-condition.p_right.to_numpy()
    np.testing.assert_allclose(condition.fitted_p_right, predicted, rtol=0, atol=2e-14)
    np.testing.assert_allclose(condition.residual, residual, rtol=0, atol=2e-14)
    assert np.isclose(row.unweighted_sse, residual @ residual, rtol=0, atol=2e-14)
    assert np.isclose(row.rmse, np.sqrt(np.mean(residual**2)), rtol=0, atol=2e-14)


# %% Saved figure marker/curve counts, dimensions, colors, and preserved inputs
artifact_checks = []
for page in manifest["pages"]:
    view, panels = page["view"], page["panels"]
    assert panels == EXPECTED_GROUPS[view]/3
    with Image.open(REPO/page["png"]) as png:
        assert png.size == EXPECTED_PIXELS[view]
        png.verify()
    with fitz.open(REPO/page["pdf"]) as pdf:
        assert pdf.page_count == 1
        text = pdf[0].get_text()
        assert sum(line.strip() == "P(right)" for line in text.splitlines()) == panels
        assert "measured RT in [0,1] s" in text
        drawings = pdf[0].get_drawings()
        for genotype, rgb in COLORS.items():
            markers = [d for d in drawings if d["fill"] is not None and np.allclose(d["fill"], rgb, rtol=0, atol=2e-6)]
            assert len(markers) == 10*panels+panels+1
            curves = [d for d in drawings if d["fill"] is None and d["color"] is not None
                and np.allclose(d["color"], rgb, rtol=0, atol=2e-6) and np.isclose(d["width"], 1.8, rtol=0, atol=2e-6)]
            assert len(curves) == int((fits.view.eq(view) & fits.genotype.eq(genotype) & fits.fit_status.eq("ok")).sum())
    assert page["xlim"] == [-18, 18] and page["ylim"] == [0, 1]
    artifact_checks.append({"view": view, "panels": panels, "pdf_pages": 1, "PNG_pixels": EXPECTED_PIXELS[view]})
for name, digest in manifest["protected_file_sha256_unchanged"].items():
    with (REPO/name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
assert hashlib.sha256((REPO/manifest["source_script"]).read_bytes()).hexdigest() == manifest["source_script_sha256"]
report = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)), "input_sha256": manifest["input_sha256"],
    "prepared_rows": prepared, "base_choices": base_choices, "selected_choices_per_view": chosen,
    "right_choices": right_choices, "left_choices": chosen-right_choices, "animals_per_view": len(chosen_animals),
    "RT_exclusions": {"nonfinite": nonfinite, "below_0": below, "above_1": above},
    "condition_points_checked": len(points), "ABL_cells_checked": len(abl_table), "memberships_checked": len(membership),
    "fits_checked": len(fits), "valid_fits": int(fits.fit_status.eq("ok").sum()), "all_six_SHANK3_cohort2_animals_retained": True,
    "checks": ["independent CSV selection/counts", "observed timed_rt window, not max_rt setting", "pooled right/all choice probabilities",
               "all 53 animals retained", "unweighted fit residuals", "saved PDF colors/marker/curve counts", "previous exports and CSVs unchanged"],
    "artifact_checks": artifact_checks}
(OUTPUT/"rt_window_independent_validation.json").write_text(json.dumps(report, indent=2)+"\n")
print("Independent verification passed: 466,978 choices, 53 animals, 210 points, 21 fits, and three PNG/PDF pairs.")
