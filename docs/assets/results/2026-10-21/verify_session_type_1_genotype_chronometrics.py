# %% Imports and parameters
"""Independently check raw RT pools, animal-level SEM, and actual PDF marks."""
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_chronometrics"
manifest = json.loads((OUTPUT / "validation.json").read_text())
ABLS = [20, 40, 60]
ILDS = [1, 2, 4, 8, 16]
GENOTYPES = ["wt", "hom", "het"]
COLORS = {"wt": "#1f77b4", "hom": "#2ca02c", "het": "#ff7f0e"}
OFFSETS = {"wt": -.12, "hom": 0, "het": .12}
assert manifest["status"] == "passed" and manifest["RT_window_s"] == [0, 1]


# %% Parse raw RTs independently; do not reuse the plotting script's grouping code
raw = defaultdict(list)
positive = Counter()
all_animals = set()
animal_identity = {}
rows_checked = 0
with (REPO / manifest["input_csv"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        rows_checked += 1
        assert row["session_type"] == "1" and row["training_level"] == "16"
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"] or 0) == 0
        assert not (row["filename"] == "CNTNAP2_cohort4.csv" and float(row["session"]) < 13)
        if row["success"] not in {"1", "-1"}:
            continue
        abl = float(row["ABL"] or "nan")
        abl = 60 if abl == 59 else abl
        ild, rt = float(row["ILD"] or "nan"), float(row["timed_rt"] or "nan")
        if abl not in ABLS or abs(ild) not in ILDS or not math.isfinite(rt) or not 0 <= rt <= 1:
            continue
        family = row["filename"].split("_")[0]
        identity = (row["filename"], family, row["genotype"])
        if row["animal"] in animal_identity:
            assert animal_identity[row["animal"]] == identity
        animal_identity[row["animal"]] = identity
        key = (row["filename"], family, row["animal"], row["genotype"], int(abl), int(abs(ild)))
        raw[key].append(rt)
        positive[key] += ild > 0
        all_animals.add(row["animal"])
assert rows_checked == 610603 and len(raw) == 795 and len(all_animals) == 53
assert sum(map(len, raw.values())) == 446949
means = {key: statistics.fmean(values) for key, values in raw.items()}
animal_audit = pd.read_csv(OUTPUT / "animal_condition_means.csv")
assert len(animal_audit) == 795
for r in animal_audit.itertuples(index=False):
    key = (r.filename, r.family, r.animal, r.genotype, r.plot_ABL, r.abs_ILD)
    values = raw[key]
    assert r.window_trials == len(values)
    assert r.positive_ild_trials == positive[key] and r.negative_ild_trials == len(values)-positive[key]
    np.testing.assert_allclose([r.animal_mean_rt_s, r.rt_sum_s], [means[key], math.fsum(values)], rtol=0, atol=3e-12)


# %% Rebuild both populations from raw per-animal values; compare all 270 estimators
expected = {}
member_counts = Counter()
group_animals = defaultdict(set)
group_trials = Counter()
for view in ["cohort", "family"]:
    conditions = defaultdict(list)
    for key in raw:
        filename, family, animal, genotype, abl, ild = key
        group = filename if view == "cohort" else family
        conditions[(view, group, genotype, abl, ild)].append(key)
        member_counts[(view, group, filename, family, genotype, animal)] += len(raw[key])
        group_animals[(view, group, genotype)].add(animal)
        group_trials[(view, group, genotype)] += len(raw[key])
    assert len(conditions) == (180 if view == "cohort" else 90)
    for condition, animal_keys in conditions.items():
        trials = sum(len(raw[key]) for key in animal_keys)
        animal_values = [means[key] for key in animal_keys]
        pooled = statistics.fmean(chain.from_iterable(raw[key] for key in animal_keys))
        n = len(animal_keys)
        average = statistics.fmean(animal_values)
        sd = statistics.stdev(animal_values) if n > 1 else math.nan
        sem = sd / math.sqrt(n)
        weighted = math.fsum(means[key]*len(raw[key]) for key in animal_keys)/trials
        assert math.isclose(pooled, weighted, rel_tol=0, abs_tol=2e-14)
        expected[condition] = {"trials": trials, "animals": n, "pooled": pooled, "animal_average": average,
            "sd": sd, "sem": sem, "positive": sum(positive[key] for key in animal_keys)}
points = pd.read_csv(OUTPUT / "chronometric_points.csv")
assert len(points) == 270
assert points.groupby("view").window_trials.sum().to_dict() == {"cohort": 446949, "family": 446949}
for r in points.itertuples(index=False):
    key = (r.view, r.group, r.genotype, r.plot_ABL, r.abs_ILD)
    e = expected[key]
    assert r.window_trials == e["trials"] and r.animals == e["animals"]
    assert r.positive_ild_trials == e["positive"] and r.negative_ild_trials == e["trials"]-e["positive"]
    np.testing.assert_allclose([r.pooled_mean_rt_s, r.animal_mean_average_rt_s], [e["pooled"], e["animal_average"]], rtol=0, atol=2e-14)
    if e["animals"] > 1:
        assert r.sem_status == "ok"
        np.testing.assert_allclose([r.across_animal_sd_rt_s, r.animal_sem_rt_s, r.mean_minus_sem_s, r.mean_plus_sem_s],
            [e["sd"], e["sem"], e["pooled"]-e["sem"], e["pooled"]+e["sem"]], rtol=0, atol=2e-14)
    else:
        assert r.sem_status == "unavailable_single_animal"
        assert all(math.isnan(x) for x in [r.across_animal_sd_rt_s, r.animal_sem_rt_s, r.mean_minus_sem_s, r.mean_plus_sem_s])
    assert math.isclose(r.plot_x, ILDS.index(r.abs_ILD)+OFFSETS[r.genotype], rel_tol=0, abs_tol=2e-15)
    assert r.color == COLORS[r.genotype]
unavailable = points.loc[points.sem_status.ne("ok")]
assert len(unavailable) == 30 and unavailable.view.eq("cohort").all()
assert unavailable.group.eq("SHANK3_cohort2.csv").all() and set(unavailable.genotype) == {"wt", "hom"}
assert points.sem_status.eq("ok").sum() == 240
members = pd.read_csv(OUTPUT / "group_membership.csv")
assert len(members) == 106
for r in members.itertuples(index=False):
    assert r.window_trials == member_counts[(r.view, r.group, r.filename, r.family, r.genotype, r.animal)]
for view, rows in members.groupby("view"):
    assert set(rows.animal) == all_animals and rows.window_trials.sum() == 446949
groups = pd.read_csv(OUTPUT / "group_counts.csv")
assert len(groups) == 18
for r in groups.itertuples(index=False):
    assert r.animals == len(group_animals[(r.view, r.group, r.genotype)])
    assert r.window_trials == group_trials[(r.view, r.group, r.genotype)]
print("Independent source statistics: 795 animal means, 270 pooled means, 240 animal SEMs, 30 single-animal cases", flush=True)


# %% Check the actual saved PDF dots, SEM endpoints, square panels, and missing bars
layout = pd.read_csv(OUTPUT / "panel_layout.csv")
plotted = pd.read_csv(OUTPUT / "plotted_points.csv")
assert len(layout) == 18 and len(plotted) == 270
reports = []
for figure in manifest["figures"]:
    view = figure["view"]
    figure_layout = layout.loc[layout.view.eq(view)]
    assert len(figure_layout) == figure["rows"]*3
    assert figure_layout.plot_ABL.tolist() == ABLS*figure["rows"]
    assert figure_layout.group.drop_duplicates().tolist() == figure["groups"]
    assert figure_layout.width_in.eq(2.45).all() and figure_layout.height_in.eq(2.45).all()
    assert figure_layout.ylim_low.eq(0).all() and figure_layout.ylim_high.eq(.45).all()
    with fitz.open(REPO / figure["pdf"]) as document:
        assert len(document) == 1
        page = document[0]
        assert math.isclose(page.rect.width, figure["width_in"]*72, abs_tol=.001)
        assert math.isclose(page.rect.height, figure["height_in"]*72, abs_tol=.001)
        text = page.get_text()
        assert "Pooled mean RT; error bars = SEM across animal means" in text
        assert ("SEM N/A" in text) == (view == "cohort")
        drawings = page.get_drawings()
        for genotype in GENOTYPES:
            color = COLORS[genotype]
            rgb = tuple(int(color[i:i+2], 16)/255 for i in [1, 3, 5])
            circles = [d for d in drawings if d["fill"] is not None and np.allclose(d["fill"], rgb, atol=1e-6)
                and any(item[0] == "c" for item in d["items"])]
            drawn = plotted.loc[plotted.view.eq(view) & plotted.genotype.eq(genotype)]
            assert len(circles) == len(drawn)+1  # Include the figure legend's circle.
            centers = np.array([[(d["rect"].x0+d["rect"].x1)/2, (d["rect"].y0+d["rect"].y1)/2] for d in circles])
            bars = []
            for d in drawings:
                if d["fill"] is None and d["color"] is not None and np.allclose(d["color"], rgb, atol=1e-6):
                    bars.extend([[item[1].x, min(item[1].y, item[2].y), max(item[1].y, item[2].y)] for item in d["items"]
                        if item[0] == "l" and abs(item[1].x-item[2].x) < 1e-4])
            bars = np.array(bars)
            for r in drawn.itertuples(index=False):
                e = expected[(view, r.group, genotype, r.plot_ABL, r.abs_ILD)]
                box = figure_layout.loc[figure_layout.group.eq(r.group) & figure_layout.plot_ABL.eq(r.plot_ABL)].iloc[0]
                plot_x = ILDS.index(r.abs_ILD)+OFFSETS[genotype]
                x = (box.x_in+(plot_x+.5)/5*box.width_in)*72
                y = (figure["height_in"]-box.y_in-e["pooled"]/.45*box.height_in)*72
                np.testing.assert_allclose([r.pdf_x_pt, r.pdf_y_pt, r.pooled_mean_rt_s], [x, y, e["pooled"]], rtol=0, atol=3e-11)
                assert r.color == color and (np.max(abs(centers-[x, y]), axis=1) < .02).sum() == 1
                if e["animals"] > 1:
                    top = (figure["height_in"]-box.y_in-(e["pooled"]+e["sem"])/.45*box.height_in)*72
                    bottom = (figure["height_in"]-box.y_in-(e["pooled"]-e["sem"])/.45*box.height_in)*72
                    np.testing.assert_allclose([r.pdf_sem_top_pt, r.pdf_sem_bottom_pt], [top, bottom], rtol=0, atol=3e-11)
                    assert len(bars) and (np.max(abs(bars-[x, top, bottom]), axis=1) < .02).sum() == 1
                    assert 0 <= e["pooled"]-e["sem"] <= e["pooled"]+e["sem"] <= .45
                else:
                    assert math.isnan(r.animal_sem_rt_s) and math.isnan(r.pdf_sem_top_pt) and math.isnan(r.pdf_sem_bottom_pt)
                    panel_top = (figure["height_in"]-box.y_in-box.height_in)*72
                    panel_bottom = (figure["height_in"]-box.y_in)*72
                    assert not len(bars) or not np.any((np.abs(bars[:, 0]-x) < .02)
                        & (bars[:, 1] >= panel_top) & (bars[:, 2] <= panel_bottom))
        for block in page.get_text("blocks"):
            assert block[0] >= 0 and block[1] >= 0 and block[2] <= page.rect.width and block[3] <= page.rect.height
    with Image.open(REPO / figure["png"]) as image:
        assert abs(image.width-figure["width_in"]*200) <= 1 and abs(image.height-figure["height_in"]*200) <= 1
        image.verify()
    reports.append({"view": view, "panels": figure["rows"]*3, "PDF_dots_checked": figure["points"],
        "PDF_error_bars_checked": figure["available_SEMs"], "PDF_pages": 1})
for relative, digest in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
report = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_rows_checked": rows_checked, "RTs_per_view": 446949, "animals_per_view": 53,
    "animal_condition_means_checked": 795, "pooled_means_checked": 270, "across_animal_SEMs_checked": 240,
    "unavailable_single_animal_SEMs_checked": 30, "memberships_checked": 106,
    "PDF_points_checked": 270, "PDF_error_bars_checked": 240, "PDF_single_animal_bars_absent": 30,
    "figure_checks": reports, "protected_files_checked": len(manifest["protected_file_sha256"]),
    "checks": ["Standard-library CSV selection and signed pooling", "Pooled means from raw RTs and trial-weighted animal means",
        "Animal SEM uses sample SD about unweighted animal-mean center", "No zero substitution for single-animal SEM",
        "Actual PDF colored dots and SEM endpoints", "Cohort/family orders and common square geometry", "Unchanged inputs/previous exports"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
