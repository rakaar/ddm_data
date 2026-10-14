# %% Imports and parameters
"""Independent valid-trial RT statistics and saved scatter/error-bar checks.

Run .venv/bin/python asd_analysis/verify_session_type_1_animal_rt_summary.py.
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
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_animal_rt_summary"
manifest = json.loads((OUTPUT / "validation.json").read_text())
ABLS = {20, 40, 60}
ILDS = {-16, -8, -4, -2, -1, 1, 2, 4, 8, 16}


# %% Read the prepared CSV with the standard library, keeping every finite RT
choice_counts, missing = Counter(), Counter()
raw_rt = defaultdict(list)
animals = {}
with (REPO / manifest["input"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        abl = float(row["ABL"])
        if abl == 59:
            abl = 60
        if float(row["success"]) not in {1, -1} or abl not in ABLS or float(row["ILD"]) not in ILDS:
            continue
        assert float(row["training_level"]) == 16 and float(row["session_type"]) == 1
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"]) == 0
        key = (row["animal"], int(abl))
        identity = (row["filename"], row["genotype"])
        assert animals.get(row["animal"], identity) == identity
        animals[row["animal"]] = identity
        choice_counts[key] += 1
        rt = float(row["timed_rt"] or "nan")
        if math.isfinite(rt):
            raw_rt[key].append(rt)
        else:
            missing[key] += 1
assert len(animals) == 53 and len(raw_rt) == 159
assert sum(choice_counts.values()) == 471416 and sum(map(len, raw_rt.values())) == 469228
assert sum(missing.values()) == 2188
print("Independent CSV selection: 469,228 finite RTs in 159 animal/ABL groups", flush=True)


# %% Python sample statistics and independently checked binomial interval coverage
table = pd.read_csv(OUTPUT / "rt_statistics.csv")
assert len(table) == 159
independent = {}
for row in table.itertuples(index=False):
    key = (row.animal, row.plot_ABL)
    rt = raw_rt[key]
    n = len(rt)
    expected = {"mean_rt_s": descriptive.fmean(rt), "sample_sd_rt_s": descriptive.stdev(rt),
                "median_rt_s": descriptive.median(rt)}
    expected["sem_rt_s"] = expected["sample_sd_rt_s"] / math.sqrt(n)
    assert row.finite_rt_trials == n and row.choice_trials == choice_counts[key]
    assert row.nonfinite_rt_trials == missing[key]
    assert (row.filename, row.genotype) == animals[row.animal]
    for name, value in expected.items():
        assert np.isclose(getattr(row, name), value, rtol=0, atol=1e-12), (key, name)
    ordered = sorted(rt)
    r, upper_rank = row.median_ci_lower_rank, row.median_ci_upper_rank
    assert upper_rank == n - r + 1
    assert np.isclose(row.median_ci_lower_s, ordered[r - 1], rtol=0, atol=1e-14)
    assert np.isclose(row.median_ci_upper_s, ordered[upper_rank - 1], rtol=0, atol=1e-14)
    # Invert the central binomial probability by checking the chosen ranks
    # and the next narrower interval, without using the plotting script's ppf.
    coverage = float(binom.cdf(n - r, n, .5) - binom.cdf(r - 1, n, .5))
    narrower_coverage = float(binom.cdf(n - r - 1, n, .5) - binom.cdf(r, n, .5))
    assert coverage >= .95 and narrower_coverage < .95
    assert np.isclose(row.median_ci_binomial_coverage, coverage, rtol=0, atol=1e-12)
    assert row.median_ci_lower_s <= expected["median_rt_s"] <= row.median_ci_upper_s
    expected.update(median_ci_lower_s=ordered[r - 1], median_ci_upper_s=ordered[upper_rank - 1])
    independent[key] = expected


# %% Membership, ordering, and all 477 plotted points and bounds
members = pd.read_csv(OUTPUT / "animal_membership.csv")
assert len(members) == 53 and members.animal.to_list() == manifest["animal_order"]
for row in members.itertuples(index=False):
    assert row.finite_rt_trials == sum(len(raw_rt[(row.animal, a)]) for a in ABLS)
    assert row.choice_trials == sum(choice_counts[(row.animal, a)] for a in ABLS)
positions = members.set_index("animal").x_position
plotted = pd.read_csv(OUTPUT / "plotted_points.csv")
assert len(plotted) == 477
for row in plotted.itertuples(index=False):
    expected = independent[(row.animal, row.plot_ABL)]
    if row.metric == "mean_sd":
        center = expected["mean_rt_s"]
        lower, upper = center - expected["sample_sd_rt_s"], center + expected["sample_sd_rt_s"]
    elif row.metric == "mean_sem":
        center = expected["mean_rt_s"]
        lower, upper = center - expected["sem_rt_s"], center + expected["sem_rt_s"]
    else:
        assert row.metric == "median_ci95"
        center, lower, upper = expected["median_rt_s"], expected["median_ci_lower_s"], expected["median_ci_upper_s"]
    np.testing.assert_allclose([row.center_s, row.lower_s, row.upper_s], [center, lower, upper], rtol=0, atol=1e-12)
    assert np.isclose(row.x, positions[row.animal] + manifest["abl_offsets"][str(row.plot_ABL)])
    assert row.color == manifest["abl_colors"][str(row.plot_ABL)]
    assert row.marker == manifest["abl_markers"][str(row.plot_ABL)]


# %% Single-page PDFs contain all animals, 159 dots and 159 vertical error bars
reports = []
for figure in manifest["figures"]:
    observed_points = plotted.loc[plotted.metric.eq(figure["metric"])]
    assert figure["ylim_s"][0] <= observed_points.lower_s.min()
    assert figure["ylim_s"][1] >= observed_points.upper_s.max()
    with fitz.open(REPO / figure["pdf"]) as document:
        assert len(document) == 1
        text = document[0].get_text()
        assert set(re.findall(r"\bASD\d{4}\b", text)) == set(members.animal)
        paths = document[0].get_drawings()
        for abl, color in manifest["abl_colors"].items():
            rgb = np.array([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
            # Matplotlib saves horizontal error-bar caps as filled narrow
            # rectangles. Separate those from circle/square/triangle markers.
            filled = [p for p in paths if p["fill"] is not None and np.allclose(p["fill"], rgb, atol=1e-6)]
            markers = sum(p["rect"].width > 0 and p["rect"].height > 0
                          and max(p["rect"].width / p["rect"].height, p["rect"].height / p["rect"].width) < 1.8
                          for p in filled)
            assert markers == 54, (figure["metric"], abl, markers)
            assert len(filled) - markers == 106  # Two caps per animal.
            verticals = 0
            for p in paths:
                if p["fill"] is not None or p["color"] is None or not np.allclose(p["color"], rgb, atol=1e-6):
                    continue
                for item in p["items"]:
                    if item[0] == "l" and abs(item[1].x - item[2].x) < 1e-4 and abs(item[1].y - item[2].y) > 1e-6:
                        verticals += 1
            assert verticals == 53, (figure["metric"], abl, verticals)
    reports.append({"metric": figure["metric"], "pdf_pages": 1, "colored_data_points": 159, "vertical_error_bars": 159})
for relative, expected in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
report = {"status": "passed", "observed": manifest["observed"], "animal_abl_groups": 159,
    "plotted_points": 477, "plotted_error_bars": 477, "figures": reports,
    "checks": ["independent standard-library CSV selection", "Python mean/stdev/median and SEM",
        "exact median order-statistic endpoints; coverage >=95% and next narrower interval <95%",
        "all animal memberships and plotted values/bounds", "saved PDF colors, dot/error-bar counts and animal IDs",
        "error bounds are fully displayed", "source CSV and existing RTD exports unchanged by SHA256"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print("Independent verification passed: three PDFs, 477 plotted estimates and error bars", flush=True)
