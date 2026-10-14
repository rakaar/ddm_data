# %% Imports and parameters
"""Independently verify pooled trial means/SEMs and all 795 PDF error bars.

Run .venv/bin/python asd_analysis/verify_all_animals_psychometric_rtd_chronometric.py.
"""
import csv
import hashlib
import json
import math
import re
import statistics as descriptive
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric"
manifest = json.loads((OUTPUT / "validation.json").read_text())
REFERENCE = REPO / manifest["reference_output"]
ABLS = {20, 40, 60}
ABS_ILDS = [1, 2, 4, 8, 16]


# %% Independent standard-library CSV selection and signed-condition counting
choice_counts, positive_choices, positive_finite, negative_finite = Counter(), Counter(), Counter(), Counter()
raw_rt, all_abl_rt = defaultdict(list), defaultdict(list)
rt_above_1 = Counter()
with (REPO / manifest["input"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        if float(row["success"]) not in {1, -1}:
            continue
        abl = float(row["ABL"])
        if abl == 59:
            abl = 60
        ild = float(row["ILD"])
        if abl not in ABLS or abs(ild) not in ABS_ILDS:
            continue
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"]) == 0
        key = (row["filename"], row["genotype"], row["animal"], int(abl), int(abs(ild)))
        choice_counts[key] += 1
        positive_choices[key] += ild > 0
        rt = float(row["timed_rt"] or "nan")
        if math.isfinite(rt):
            raw_rt[key].append(rt)
            all_abl_rt[(row["animal"], int(abl))].append(rt)
            positive_finite[key] += ild > 0
            negative_finite[key] += ild < 0
            rt_above_1[key] += rt > 1
assert len(raw_rt) == 795 and len(all_abl_rt) == 159
assert sum(choice_counts.values()) == 471416 and sum(map(len, raw_rt.values())) == 469228
assert sum(rt_above_1.values()) == 2250
print("Independent CSV selection: 795 conditions; 469,228 finite valid-trial RTs", flush=True)


# %% Combined-trial sample statistics, both signs, and ABL totals
chrono = pd.read_csv(OUTPUT / "chronometric_points.csv")
assert len(chrono) == 795 and chrono.animal.nunique() == 53
for row in chrono.itertuples(index=False):
    key = (row.filename, row.genotype, row.animal, row.plot_ABL, row.abs_ILD)
    rt = raw_rt[key]
    n = len(rt)
    expected_mean = descriptive.fmean(rt)
    expected_sd = descriptive.stdev(rt)
    expected_sem = expected_sd / math.sqrt(n)
    assert row.valid_choice_trials == choice_counts[key] and row.finite_rt_trials == n
    assert row.nonfinite_rt_trials == choice_counts[key] - n
    assert row.positive_ild_choice_trials == positive_choices[key]
    assert row.negative_ild_choice_trials == choice_counts[key] - positive_choices[key]
    assert row.positive_ild_rt_trials == positive_finite[key] and row.negative_ild_rt_trials == negative_finite[key]
    assert row.positive_ild_rt_trials + row.negative_ild_rt_trials == n
    assert row.rt_above_1_s == rt_above_1[key]
    np.testing.assert_allclose([row.mean_rt_s, row.sample_sd_rt_s, row.sem_rt_s],
                              [expected_mean, expected_sd, expected_sem], rtol=0, atol=1e-12)
    assert np.isclose(row.plot_x, ABS_ILDS.index(row.abs_ILD) + manifest["chronometric_offsets"][str(row.plot_ABL)])
    assert 0 <= row.mean_minus_sem_s <= row.mean_rt_s <= row.mean_plus_sem_s <= .45
means = pd.read_csv(REPO / manifest["reference_rt_means"]).set_index(["animal", "plot_ABL"])
for (animal, abl), rows in chrono.groupby(["animal", "plot_ABL"]):
    weighted_mean = np.average(rows.mean_rt_s, weights=rows.finite_rt_trials)
    expected = descriptive.fmean(all_abl_rt[(animal, abl)])
    assert np.isclose(weighted_mean, expected, rtol=0, atol=1e-12)
    assert np.isclose(weighted_mean, means.loc[(animal, abl), "mean_rt_s"], rtol=0, atol=1e-12)
    assert rows.finite_rt_trials.sum() == means.loc[(animal, abl), "finite_rt_trials"]


# %% Reference psychometric points/fits, histograms, and page assignments are unchanged
for name in manifest["copied_reference_audits"]:
    assert (OUTPUT / name).read_bytes() == (REFERENCE / name).read_bytes(), name
members = pd.read_csv(OUTPUT / "animal_pages.csv")
bins = pd.read_csv(OUTPUT / "rtd_histogram_bins.csv")
fits = pd.read_csv(OUTPUT / "psychometric_fits.csv")
plotted = pd.read_csv(OUTPUT / "chronometric_plotted_points.csv")
assert len(plotted) == 795 and len(members) == 53
for (animal, abl), group in bins.groupby(["animal", "plot_ABL"]):
    group = group.sort_values("bin_index")
    edges = np.r_[group.bin_left_s.to_numpy(), group.bin_right_s.iloc[-1]]
    counts, _ = np.histogram(all_abl_rt[(animal, abl)], bins=edges)
    np.testing.assert_array_equal(counts, group.bin_count)
    np.testing.assert_allclose(np.diff(edges), .020, rtol=0, atol=1e-12)
    np.testing.assert_allclose(group.density_per_s, counts / (len(all_abl_rt[(animal, abl)]) * .020), rtol=0, atol=1e-12)
    assert np.isclose(np.dot(group.density_per_s, np.diff(edges)), 1, rtol=0, atol=1e-12)
lookup = chrono.set_index(["animal", "plot_ABL", "abs_ILD"])
for point in plotted.itertuples(index=False):
    row = lookup.loc[(point.animal, point.plot_ABL, point.abs_ILD)]
    np.testing.assert_allclose([point.mean_rt_s, point.sem_rt_s, point.lower_s, point.upper_s, point.plot_x],
        [row.mean_rt_s, row.sem_rt_s, row.mean_minus_sem_s, row.mean_plus_sem_s, row.plot_x], rtol=0, atol=1e-12)
    assert point.marker == "o" and point.color == manifest["colors"][str(point.plot_ABL)]


# %% Verify the actual dots and SEM stems in PDF point coordinates
page_reports = []
old_manifest = json.loads((REFERENCE / "validation.json").read_text())
with fitz.open(REPO / old_manifest["pdf"]) as old_document:
    old_toc = old_document.get_toc()
with fitz.open(REPO / manifest["pdf"]) as document:
    assert len(document) == 11 and document.get_toc() == old_toc
    for page_number, page in enumerate(document, start=1):
        batch = members.loc[members.pdf_page.eq(page_number)]
        assert len(batch) == (5 if page_number < 11 else 3)
        layout = manifest["pages"][page_number - 1]
        assert np.isclose(page.rect.width, layout["page_width_in"] * 72)
        np.testing.assert_allclose(np.array(layout["panel_widths_in"]) / layout["panel_widths_in"][0],
                                   manifest["column_width_ratios"])
        assert np.isclose(layout["panel_heights_in"][2] / layout["panel_widths_in"][2], 1)
        text = page.get_text()
        assert set(re.findall(r"\bASD\d{4}\b", text)) == set(batch.animal)
        assert sum(line.strip().endswith("| Chronometric") for line in text.splitlines()) == len(batch)
        paths = page.get_drawings()
        page_points = plotted.loc[plotted.pdf_page.eq(page_number)]
        assert len(page_points) == len(batch) * 15
        for abl, color in manifest["colors"].items():
            abl = int(abl)
            rgb = np.array([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
            circles = [d for d in paths if d["fill"] is not None and np.allclose(d["fill"], rgb, atol=1e-6)
                       and any(item[0] == "c" for item in d["items"])]
            centers = np.array([[(d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2] for d in circles])
            verticals = []
            for drawing in paths:
                if drawing["fill"] is None and drawing["color"] is not None and np.allclose(drawing["color"], rgb, atol=1e-6):
                    for item in drawing["items"]:
                        if item[0] == "l" and abs(item[1].x - item[2].x) < 1e-4:
                            verticals.append([item[1].x, min(item[1].y, item[2].y), max(item[1].y, item[2].y)])
            verticals = np.array(verticals)
            for point in page_points.loc[page_points.plot_ABL.eq(abl)].itertuples(index=False):
                target_dot = [point.pdf_x_pt, point.pdf_mean_y_pt]
                assert (np.max(np.abs(centers - target_dot), axis=1) < .02).sum() == 1
                target_bar = [point.pdf_x_pt, point.pdf_upper_y_pt, point.pdf_lower_y_pt]
                assert (np.max(np.abs(verticals - target_bar), axis=1) < .02).sum() == 1
            expected_fits = fits.animal.isin(batch.animal) & fits.plot_ABL.eq(abl) & fits.fit_status.eq("ok")
            actual_fits = sum(d["fill"] is None and d["color"] is not None and np.allclose(d["color"], rgb, atol=1e-6)
                             and np.isclose(d["width"], 1.7, atol=1e-4) for d in paths)
            assert actual_fits == expected_fits.sum()
        page_reports.append({"page": page_number, "animals": batch.animal.to_list(),
                             "chronometric_dots_and_sem_bars_checked": len(page_points)})
for relative, expected_hash in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash, relative
report = {"status": "passed", "animals": 53, "pdf_pages": 11, "chronometric_points": 795,
    "finite_rt_trials": 469228, "rt_above_1_s_retained": 2250, "pages": page_reports,
    "checks": ["independent standard-library CSV selection; both signs and sessions pooled by observed trials",
        "Python fmean/stdev/SEM for all 795 conditions", "condition counts and weighted means reproduce animal/ABL means",
        "all psychometric fits, points, RTD bins/densities, and memberships unchanged byte-for-byte",
        "RTD counts/densities checked against raw finite RTs", "all 795 colored dots and both SEM endpoints found in saved PDF",
        "shared axes cover every error bound; categorical offsets checked", "53 original animal bookmarks; 11 widened pages",
        "prepared CSV and original native/published two-column exports preserved by SHA256"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print("Independent verification passed: 795 means/SEMs and their PDF dots/error bars; 53 animals, 11 pages", flush=True)
