# %% Imports and parameters
"""Independently parse choices/RTs and verify the 11-page, 53-animal export.

Run .venv/bin/python asd_analysis/verify_all_animals_psychometric_and_rtd.py.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd"
manifest = json.loads((OUTPUT / "validation.json").read_text())
INPUT = REPO / manifest["input"]
ABLS = {20, 40, 60}
ILDS = {-16, -8, -4, -2, -1, 1, 2, 4, 8, 16}
BIN_WIDTH = .020


# %% Independent standard-library CSV read, without a measured-RT cutoff
trials, rights, member_trials, member_rights, prepared = Counter(), Counter(), Counter(), Counter(), Counter()
reaction_times = defaultdict(list)
choice_rt_missing = Counter()
total_rows = 0
with INPUT.open(newline="") as stream:
    for row in csv.DictReader(stream):
        total_rows += 1
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"]) == 0
        prepared[row["animal"]] += 1
        abl = float(row["ABL"])
        if abl == 59:
            abl = 60
        ild = float(row["ILD"])
        if float(row["success"]) not in {1, -1} or abl not in ABLS or ild not in ILDS:
            continue
        animal, genotype, filename = row["animal"], row["genotype"], row["filename"]
        identity = (filename, genotype, animal)
        key = (*identity, int(abl), ild)
        side = float(row["response_poke"])
        assert side in {-1, 1}
        trials[key] += 1
        rights[key] += side == 1
        member_trials[identity] += 1
        member_rights[identity] += side == 1
        rt = float(row["timed_rt"] or "nan")
        if np.isfinite(rt):
            reaction_times[(animal, int(abl))].append(rt)
        else:
            choice_rt_missing[(animal, int(abl))] += 1
assert total_rows == 642370 and len(member_trials) == 53
assert sum(trials.values()) == 471416 and sum(rights.values()) == 241539
assert sum(map(len, reaction_times.values())) == 469228 and sum(choice_rt_missing.values()) == 2188
assert sum(sum(rt > 1 for rt in values) for values in reaction_times.values()) == 2250
print("Independent CSV parsing passed: 53 animals, 471,416 choices, 469,228 finite RTs", flush=True)


# %% Every animal and measured probability point agrees with the independent counts
members = pd.read_csv(OUTPUT / "animal_pages.csv")
points = pd.read_csv(OUTPUT / "psychometric_points.csv")
fits = pd.read_csv(OUTPUT / "psychometric_fits.csv")
counts = pd.read_csv(OUTPUT / "abl_counts.csv")
bins = pd.read_csv(OUTPUT / "rtd_histogram_bins.csv")
assert len(members) == 53 and len(points) == 1590 and len(fits) == len(counts) == 159
for member in members.itertuples(index=False):
    identity = (member.filename, member.genotype, member.animal)
    assert member.choice_trials == member_trials[identity]
    assert member.right_trials == member_rights[identity]
    assert member.left_trials == member.choice_trials - member.right_trials
    assert member.prepared_animal_rows == prepared[member.animal]
    n_finite = sum(len(reaction_times[(member.animal, abl)]) for abl in ABLS)
    assert member.finite_rt_trials == n_finite and member.nonfinite_rt_trials == member.choice_trials - n_finite
for point in points.itertuples(index=False):
    key = (point.filename, point.genotype, point.animal, point.plot_ABL, point.ILD)
    assert point.trials == trials[key] and point.right_trials == rights[key]
    assert point.left_trials == point.trials - point.right_trials
    assert np.isclose(point.p_right, rights[key] / trials[key], rtol=0, atol=1e-14)
reference = pd.read_csv(REPO / manifest["fit_source"])
reference = reference.loc[reference.view.eq("animal")].set_index(["group", "plot_ABL"])
for fit in fits.itertuples(index=False):
    old = reference.loc[(fit.group, fit.plot_ABL)]
    np.testing.assert_allclose([fit.upper, fit.lower, fit.x0, fit.k], old[["upper", "lower", "x0", "k"]].to_numpy(dtype=float), rtol=0, atol=1e-14)
    assert fit.fit_status == old.fit_status
    p = points.loc[points.animal.eq(fit.animal) & points.plot_ABL.eq(fit.plot_ABL)].sort_values("ILD")
    assert p.trials.sum() == fit.trials and p.right_trials.sum() == fit.right_trials
    if fit.fit_status == "ok":
        fitted = fit.lower + (fit.upper - fit.lower) * expit(fit.k * (p.ILD.to_numpy() - fit.x0))
        np.testing.assert_allclose(p.fitted_p_right, fitted, rtol=0, atol=1e-12)


# %% Verify full-RT histograms, 20 ms widths, normalizers, and area 1
for (animal, abl), b in bins.groupby(["animal", "plot_ABL"]):
    b = b.sort_values("bin_index")
    edges = np.r_[b.bin_left_s.to_numpy(), b.bin_right_s.iloc[-1]]
    rt = np.array(reaction_times[(animal, abl)])
    expected_counts, _ = np.histogram(rt, bins=edges)
    assert expected_counts.sum() == len(rt)
    np.testing.assert_array_equal(b.bin_count, expected_counts)
    np.testing.assert_allclose(np.diff(edges), BIN_WIDTH, rtol=0, atol=1e-12)
    assert b.normalization_trials.eq(len(rt)).all()
    np.testing.assert_allclose(b.density_per_s, expected_counts / (len(rt) * BIN_WIDTH), rtol=0, atol=1e-12)
    assert np.isclose(np.dot(b.density_per_s, np.diff(edges)), 1, rtol=0, atol=1e-12)
    c = counts.loc[counts.animal.eq(animal) & counts.plot_ABL.eq(abl)].iloc[0]
    assert c.finite_rt_trials == c.histogram_trials == len(rt)
    assert c.nonfinite_rt_trials == choice_rt_missing[(animal, abl)]
    assert c.rt_above_1_s == (rt > 1).sum()
    assert c.rt_in_display == ((rt >= 0) & (rt <= .6)).sum()
# ASD0054's earlier single-animal RT histogram is preserved exactly.
single = pd.read_csv(REPO / "asd_analysis/session_type_1_animal_psychometric_rtd/ASD0054/rtd_histogram_bins.csv")
pd.testing.assert_frame_equal(single.set_index(["plot_ABL", "bin_index"])[["bin_count", "density_per_s"]],
                              bins.loc[bins.animal.eq("ASD0054")].set_index(["plot_ABL", "bin_index"])[["bin_count", "density_per_s"]])


# %% Verify all 11 PDF pages, animal identities, colors, points, and fit curves
page_reports = []
with fitz.open(REPO / manifest["pdf"]) as document:
    assert len(document) == 11
    assert len([entry for entry in document.get_toc() if entry[0] == 3]) == 53
    for page_number, page in enumerate(document, start=1):
        batch = members.loc[members.pdf_page.eq(page_number)]
        n = len(batch)
        assert n == (5 if page_number < 11 else 3)
        text = page.get_text()
        assert set(re.findall(r"\bASD\d{4}\b", text)) == set(batch.animal)
        assert text.count("0.6") >= n
        drawings = page.get_drawings()
        for abl, color in manifest["colors"].items():
            abl = int(abl)
            rgb = np.array([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
            markers = sum(d["fill"] is not None and np.allclose(d["fill"], rgb, atol=1e-6) for d in drawings)
            assert markers == n * 11 + 1, (page_number, abl, markers)
            fit_paths = sum(d["fill"] is None and d["color"] is not None and np.allclose(d["color"], rgb, atol=1e-6)
                            and np.isclose(d["width"], 1.7, atol=1e-4) for d in drawings)
            expected_fits = fits.animal.isin(batch.animal) & fits.plot_ABL.eq(abl) & fits.fit_status.eq("ok")
            assert fit_paths == expected_fits.sum(), (page_number, abl, fit_paths)
        page_reports.append({"page": page_number, "animals": batch.animal.to_list(), "rows": n,
                             "colors_markers_and_fit_curves": "passed"})
for relative, expected_hash in manifest["protected_file_sha256"].items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash, relative
report = {"status": "passed", "observed": manifest["observed"], "psychometric_points": 1590,
    "fit_status_counts": fits.fit_status.value_counts().to_dict(), "rtd_curves": 159, "pdf_pages": 11,
    "pages": page_reports, "checks": ["independent CSV parsing, all memberships and point counts",
        "reused fit parameters and predictions", "every full-RT histogram against raw finite values",
        "20 ms bin widths and full-range unit density areas", "ASD0054 histogram unchanged",
        "53 animal bookmarks, page identities, PDF colors, scatter counts and fit paths",
        "source CSV and previous ASD0054 artifacts unchanged by SHA256"]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Independent verification passed: {len(members)} animals, 11 pages, 159 full-RT densities", flush=True)
