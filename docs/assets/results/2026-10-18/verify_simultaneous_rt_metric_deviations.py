"""Independent trial, fit, arithmetic, and rendered-figure checks.

Run with .venv/bin/python ASD_Human_Data/verify_simultaneous_rt_metric_deviations.py.
"""

# %% Imports and inputs
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq, curve_fit

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "simultaneous_rt_metric_deviations"
EXTRACTION = json.loads((HERE / "simultaneous_rt_extraction.json").read_text())
MANIFEST = json.loads((OUTPUT / "validation.json").read_text())
ASLS = (20, 40)
DELTAS = (-2, -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2)
assert hashlib.sha256((HERE / "simultaneous_rt_trials.csv").read_bytes()).hexdigest() == EXTRACTION["output_sha256"] == MANIFEST["source_sha256"]
assert EXTRACTION["selection"] == "Condition == 'RT'"


# %% Reconstruct the 380 measured probabilities from individual choices
cells = defaultdict(Counter)
membership = {}
sessions = defaultdict(set)
with (HERE / "simultaneous_rt_trials.csv").open(newline="") as stream:
    for trial in csv.DictReader(stream):
        participant = int(trial["Participant"])
        group = trial["Group"]
        asl = int(float(trial["ASL_cd"]))
        delta = round(float(trial["Delta_dB"]), 2)
        choice = int(trial["Choice"])
        assert trial["Condition"] == "RT" and int(trial["Valid"]) == 1
        assert not trial["Stim_duration_s"] and float(trial["RT_s"]) > 0
        assert choice in [-1, 1] and int(trial["ChoseRight"]) == int(choice == 1)
        assert group in ["Control", "ASD"] and int(trial["GroupID"]) == {"Control": 0, "ASD": 1}[group]
        assert asl in ASLS and delta in DELTAS
        assert math.isclose(float(trial["Stim_right_dB"]) - float(trial["Stim_left_dB"]),
                            float(trial["Delta_dB"]), abs_tol=1e-12)
        if participant in membership:
            assert membership[participant] == group
        membership[participant] = group
        sessions[participant].add(int(trial["Session"]))
        cell = cells[(participant, asl, delta)]
        cell["trials"] += 1
        cell["right"] += choice == 1
        cell["left"] += choice == -1

order = sorted((p for p in membership if membership[p] == "Control")) + sorted(
    (p for p in membership if membership[p] == "ASD"))
assert len(cells) == 380 and sum(c["trials"] for c in cells.values()) == 11400
assert len(order) == 19 and len(order[:11]) == 11 and all(membership[p] == "Control" for p in order[:11])
assert all(membership[p] == "ASD" for p in order[11:])
assert all(len(sessions[p]) == 2 for p in order)
assert all(c["trials"] == 30 for c in cells.values())
assert sum(c["right"] for c in cells.values()) == 5883
assert sum(c["left"] for c in cells.values()) == 5517
assert MANIFEST["participant_order"] == order and MANIFEST["choices"] == 11400

condition_counts = pd.read_csv(OUTPUT / "condition_counts.csv")
assert len(condition_counts) == 380
for row in condition_counts.itertuples(index=False):
    cell = cells[(row.Participant, row.ASL_cd, row.delta_key)]
    assert (row.trials, row.right_trials, row.left_trials) == (
        cell["trials"], cell["right"], cell["left"])
    assert math.isclose(row.p_right, cell["right"] / 30, abs_tol=2e-16)


# %% Refit the raw counts and solve JND crossings numerically
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


fits = pd.read_csv(OUTPUT / "fits.csv")
assert len(fits) == 38 and fits.fit_status.eq("ok").all() and fits.jnd_status.eq("ok").all()
crossings = 0
for row in fits.itertuples(index=False):
    x = np.array(DELTAS, dtype=float)
    probability = np.array([cells[(row.Participant, row.ASL_cd, delta)]["right"] / 30
                            for delta in DELTAS])
    calculated, _ = curve_fit(
        sigmoid, x, probability,
        p0=[probability.max(), probability.min(), np.median(x), .1],
        bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]), maxfev=10000)
    saved = np.array([row.upper, row.lower, row.x0, row.k])
    np.testing.assert_allclose(saved, calculated, rtol=0, atol=1e-10)
    upper, lower, x0, k = saved
    assert upper > lower and k > 0
    assert (row.trials, row.right_trials, row.left_trials) == (
        300, sum(cells[(row.Participant, row.ASL_cd, d)]["right"] for d in DELTAS),
        sum(cells[(row.Participant, row.ASL_cd, d)]["left"] for d in DELTAS))
    high, low = .75 * upper, .25 * (1 - lower)
    assert lower < low < high < upper
    np.testing.assert_allclose([row.y_high, row.y_low], [high, low], rtol=0, atol=1e-15)
    roots = [brentq(lambda delta: sigmoid(delta, *saved) - target,
                    x0 - 100 / k, x0 + 100 / k, xtol=1e-12) for target in [high, low]]
    np.testing.assert_allclose([row.delta_high, row.delta_low, row.jnd],
                               [roots[0], roots[1], (roots[0] - roots[1]) / 2],
                               rtol=0, atol=5e-11)
    np.testing.assert_allclose(
        condition_counts.loc[(condition_counts.Participant == row.Participant) &
                             (condition_counts.ASL_cd == row.ASL_cd)]
        .sort_values("delta_key").fitted_p_right.to_numpy(),
        sigmoid(x, *saved), rtol=0, atol=1e-12)
    crossings += 2
assert crossings == 76


# %% Rebuild each metric's eligibility, equal-ASL means and signed deltas
eligibility = pd.read_csv(OUTPUT / "eligibility.csv")
means = pd.read_csv(OUTPUT / "participant_means.csv")
values = pd.read_csv(OUTPUT / "values_and_deviations.csv")
summary = pd.read_csv(OUTPUT / "summary_statistics.csv")
group_summary = pd.read_csv(OUTPUT / "group_summary_statistics.csv")
plotted = pd.read_csv(OUTPUT / "plotted_points.csv")
mean_marks = pd.read_csv(OUTPUT / "plotted_mean_marks.csv")
bars = pd.read_csv(OUTPUT / "plotted_summary_bars.csv")
assert (len(eligibility), len(means), len(values), len(summary), len(group_summary), len(plotted), len(mean_marks), len(bars)) == (38, 38, 76, 4, 8, 152, 38, 8)
for metric, field in [("jnd", "jnd"), ("slope", "k")]:
    eligible = []
    calculated_means, calculated_deviations = [], []
    metric_eligibility = eligibility.loc[eligibility.metric == metric]
    metric_means = means.loc[means.metric == metric]
    assert metric_eligibility.Participant.tolist() == order
    assert metric_means.Participant.tolist() == order
    for position, participant in enumerate(order):
        own = fits.loc[fits.Participant == participant].set_index("ASL_cd")
        pair = [float(own.loc[asl, field]) for asl in ASLS]
        audit = metric_eligibility.loc[metric_eligibility.Participant == participant].iloc[0]
        status_field = "jnd_status" if metric == "jnd" else "fit_status"
        complete = (all(own.loc[asl, status_field] == "ok" for asl in ASLS)
                    and all(math.isfinite(v) and v > 0 for v in pair))
        assert bool(audit.eligible) == complete
        assert audit.position == position and audit.Group == membership[participant]
        np.testing.assert_allclose([audit.value_ASL_20, audit.value_ASL_40], pair, rtol=0, atol=1e-14)
        if not audit.eligible:
            continue
        eligible.append(participant)
        center = statistics.fmean(pair)
        calculated_means.append(center)
        own_mean = metric_means.loc[metric_means.Participant == participant].iloc[0]
        assert own_mean.position == position and own_mean.Group == membership[participant]
        assert own_mean.choice_trials == 600
        assert math.isclose(own_mean.participant_mean, center, abs_tol=2e-14)
        for asl, value in zip(ASLS, pair):
            deviation = value - center
            calculated_deviations.append(deviation)
            point = values.loc[(values.metric == metric) & (values.Participant == participant) &
                               (values.ASL_cd == asl)].iloc[0]
            assert point.position == position and point.Group == membership[participant]
            assert point.trials == 300
            np.testing.assert_allclose([point.value, point.participant_mean, point.signed_delta],
                                       [value, center, deviation], rtol=0, atol=2e-14)
        assert math.isclose(sum(calculated_deviations[-2:]), 0, abs_tol=2e-14)
    assert len(eligible) == 19 and MANIFEST["eligible_participants"][metric] == 19
    for level, sample in [("participant_means", calculated_means),
                          ("ASL_deviations", calculated_deviations)]:
        record = summary.loc[(summary.metric == metric) & (summary.level == level)].iloc[0]
        center, spread = statistics.fmean(sample), statistics.pstdev(sample)
        assert record.n == len(sample) and record.ddof == 0
        np.testing.assert_allclose(
            [record["mean"], record.population_sd, record.lower, record.upper],
            [center, spread, center - spread, center + spread], rtol=0, atol=2e-14)
        for group in ["Control", "ASD"]:
            if level == "participant_means":
                group_sample = [calculated_means[i] for i, p in enumerate(eligible)
                                if membership[p] == group]
            else:
                group_sample = [calculated_deviations[2*i + j]
                                for i, p in enumerate(eligible) if membership[p] == group
                                for j in range(2)]
            group_record = group_summary.loc[(group_summary.metric == metric) &
                                             (group_summary.Group == group) &
                                             (group_summary.level == level)].iloc[0]
            group_center, group_spread = statistics.fmean(group_sample), statistics.pstdev(group_sample)
            assert group_record.n == len(group_sample) and group_record.ddof == 0
            np.testing.assert_allclose(
                [group_record["mean"], group_record.population_sd,
                 group_record.lower, group_record.upper],
                [group_center, group_spread, group_center - group_spread,
                 group_center + group_spread], rtol=0, atol=2e-14)
    assert abs(statistics.fmean(calculated_deviations)) < 1e-14


# %% Inspect rendered PDF markers, summary bars, labels, and unclipped ranges
figure_checks = []
for figure in MANIFEST["figures"]:
    metric = figure["metric"]
    metric_points = values.loc[values.metric == metric]
    metric_plotted = plotted.loc[plotted.metric == metric]
    with Image.open(HERE / figure["png"]) as image:
        assert image.size == (3600, 1800)
        image.verify()
    with fitz.open(HERE / figure["pdf"]) as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        assert np.isclose(page.rect.width, 20 * 72) and np.isclose(page.rect.height, 10 * 72)
        pdf_text = page.get_text()
        text_blocks = page.get_text("blocks")
        assert "Control (n=11)" in pdf_text and "ASD (n=8)" in pdf_text
        assert "Signed deviation:" in pdf_text and "Group mean ±1 SD" in pdf_text
        for participant in order:
            assert f"P{participant}" in pdf_text
        paths = page.get_drawings()
        metric_marks = mean_marks.loc[mean_marks.metric == metric]
        assert len(metric_marks) == 19
        for group, color in MANIFEST["mean_mark_colors"].items():
            rgb = tuple(int(color[i:i+2], 16) / 255 for i in [1, 3, 5])
            mean_paths = [path for path in paths if path["fill"] is not None and
                          path["color"] is not None and np.isclose(path["width"], 1.5)
                          and np.allclose(path["fill"], rgb, atol=1e-6)
                          and np.allclose(path["color"], rgb, atol=1e-6)]
            group_marks = metric_marks.loc[metric_marks.Group == group]
            assert len(mean_paths) == len(group_marks) == {"Control": 11, "ASD": 8}[group]
            centers = np.array([[(path["rect"].x0 + path["rect"].x1) / 2,
                                 (path["rect"].y0 + path["rect"].y1) / 2]
                                for path in mean_paths])
            for mark in group_marks.itertuples(index=False):
                source = means.loc[(means.metric == metric) &
                                   (means.Participant == mark.Participant)].iloc[0]
                assert mark.marker == "_" and mark.color == color
                assert mark.Group == membership[mark.Participant]
                assert mark.x == order.index(mark.Participant)
                assert math.isclose(mark.y, source.participant_mean, abs_tol=2e-14)
                assert (np.max(abs(centers - [mark.pdf_x_pt, mark.pdf_y_pt]), axis=1) < .04).sum() == 1
        for asl, color in MANIFEST["colors"].items():
            asl = int(asl)
            rgb = tuple(int(color[i:i+2], 16) / 255 for i in [1, 3, 5])
            colored = [path for path in paths if path["fill"] is not None and
                       np.allclose(path["fill"], rgb, atol=1e-6)]
            centers = np.array([[(path["rect"].x0 + path["rect"].x1) / 2,
                                 (path["rect"].y0 + path["rect"].y1) / 2]
                                for path in colored])
            assert len(centers) >= 38
            for point in metric_plotted.loc[metric_plotted.ASL_cd == asl].itertuples(index=False):
                record = metric_points.loc[(metric_points.Participant == point.Participant) &
                                           (metric_points.ASL_cd == asl)].iloc[0]
                assert point.Group == membership[point.Participant]
                assert point.marker == {"Control": "o", "ASD": "*"}[point.Group]
                assert point.color == color
                assert math.isclose(point.x, order.index(point.Participant) + (-.105 if asl == 20 else .105), abs_tol=1e-14)
                assert math.isclose(point.y, record.value if point.panel == "raw" else record.signed_delta, abs_tol=2e-14)
                # A Matplotlib star's bounding box is about 0.5 pt above its anchor.
                tolerance = .55 if point.Group == "ASD" else .04
                assert (np.max(abs(centers - [point.pdf_x_pt, point.pdf_y_pt]), axis=1) < tolerance).sum() == 1
        pdf_bars = {}
        for group, color in MANIFEST["mean_mark_colors"].items():
            rgb = tuple(int(color[i:i+2], 16) / 255 for i in [1, 3, 5])
            colored_bars = [path for path in paths if path["fill"] is None and
                            path["color"] is not None and np.isclose(path["width"], 5)
                            and np.allclose(path["color"], rgb, atol=1e-6)]
            assert len(colored_bars) == 2
            pdf_bars[group] = []
            for path in colored_bars:
                for segment in path["items"]:
                    if segment[0] == "l":
                        pdf_bars[group].append([segment[1].x,
                                                min(segment[1].y, segment[2].y),
                                                max(segment[1].y, segment[2].y)])
            assert len(pdf_bars[group]) == 2
        for bar in bars.loc[bars.metric == metric].itertuples(index=False):
            assert bar.color == MANIFEST["mean_mark_colors"][bar.Group]
            assert bar.x == {"Control": .28, "ASD": .72}[bar.Group]
            assert (np.max(abs(np.array(pdf_bars[bar.Group]) -
                               [bar.pdf_x_pt, bar.pdf_y_top_pt, bar.pdf_y_bottom_pt]),
                           axis=1) < .04).any()
            level = "participant_means" if bar.panel == "raw" else "ASL_deviations"
            record = group_summary.loc[(group_summary.metric == metric) &
                                       (group_summary.Group == bar.Group) &
                                       (group_summary.level == level)].iloc[0]
            np.testing.assert_allclose([bar.n, bar.mean, bar.sd, bar.lower, bar.upper],
                                       [record.n, record["mean"], record.population_sd, record.lower, record.upper],
                                       rtol=0, atol=2e-14)
            shown_mean = 0.0 if abs(bar.mean) < .0005 else bar.mean
            assert bar.label == f"{bar.Group}\nmean {shown_mean:.3f}\nSD {bar.sd:.3f}"
            assert bar.label in pdf_text
            mean_hits = page.search_for(f"mean {shown_mean:.3f}")
            sd_hits = page.search_for(f"SD {bar.sd:.3f}")
            aligned_means = [rect for rect in mean_hits if abs((rect.x0 + rect.x1) / 2 - bar.pdf_x_pt) < 1]
            aligned_sds = [rect for rect in sd_hits if abs((rect.x0 + rect.x1) / 2 - bar.pdf_x_pt) < 1]
            assert len(aligned_means) == len(aligned_sds) == 1
            assert aligned_means[0].y1 < bar.pdf_y_top_pt
            assert aligned_sds[0].y1 < bar.pdf_y_top_pt
        assert figure["raw_ylim"][0] <= metric_points.value.min()
        assert figure["raw_ylim"][1] >= metric_points.value.max()
        assert figure["delta_ylim"][0] <= metric_points.signed_delta.min()
        assert figure["delta_ylim"][1] >= metric_points.signed_delta.max()
        for level, limits in [("participant_means", figure["raw_ylim"]),
                              ("ASL_deviations", figure["delta_ylim"])]:
            record = summary.loc[(summary.metric == metric) & (summary.level == level)].iloc[0]
            assert limits[0] <= record.lower and limits[1] >= record.upper
            for record in group_summary.loc[(group_summary.metric == metric) &
                                            (group_summary.level == level)].itertuples(index=False):
                assert limits[0] <= record.lower and limits[1] >= record.upper
        for block in text_blocks:
            assert block[0] >= 0 and block[1] >= 0
            assert block[2] <= page.rect.width and block[3] <= page.rect.height
    figure_checks.append({"metric": metric, "eligible_participants": 19,
                          "pdf_data_marks_checked": 76, "pdf_group_mean_marks_checked": 19,
                          "pdf_group_sd_bars_checked": 4, "bar_text_labels_checked": 4,
                          "png_pixels": [3600, 1800], "pdf_pages": 1})

report = {"status": "passed", "source_sha256": MANIFEST["source_sha256"],
          "choices_checked": 11400, "condition_probabilities_checked": 380,
          "refits_checked": 38, "numeric_jnd_crossings_checked": crossings,
          "eligible_participants": {"jnd": 19, "slope": 19},
          "participant_means_checked": 38, "metric_values_and_deltas_checked": 76,
          "saved_pdf_data_marks_checked": 152, "saved_pdf_group_mean_marks_checked": 38,
          "group_summary_rows_checked": 8, "saved_pdf_group_sd_bars_checked": 8,
          "bar_text_labels_checked": 8,
          "figure_checks": figure_checks}
(OUTPUT / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
