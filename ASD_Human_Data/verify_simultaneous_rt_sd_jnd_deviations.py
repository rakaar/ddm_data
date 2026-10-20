"""Independently verify the four-row RT/SD participant JND figure."""

# %% Inputs and source-row denominators
import hashlib
import json
from pathlib import Path
import statistics
import warnings

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq, curve_fit

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "simultaneous_rt_sd_jnd_deviations"
manifest = json.loads((OUTPUT / "validation.json").read_text())
counts = pd.read_csv(OUTPUT / "condition_counts.csv", float_precision="round_trip")
fits = pd.read_csv(OUTPUT / "fits_and_jnd_crossings.csv", float_precision="round_trip")
eligibility = pd.read_csv(OUTPUT / "eligibility.csv")
values = pd.read_csv(OUTPUT / "values_and_deviations.csv", float_precision="round_trip")
means = pd.read_csv(OUTPUT / "participant_means.csv", float_precision="round_trip")
summary = pd.read_csv(OUTPUT / "summary_statistics.csv")
groups = pd.read_csv(OUTPUT / "group_summary_statistics.csv")
asl_groups = pd.read_csv(OUTPUT / "group_asl_summary_statistics.csv")
points = pd.read_csv(OUTPUT / "plotted_points.csv")
bars = pd.read_csv(OUTPUT / "plotted_group_bars.csv")
limits = pd.read_csv(OUTPUT / "panel_limits.csv")
order = pd.read_csv(OUTPUT / "participant_order.csv")
assert len(counts) == 760 and len(fits) == 76 and len(eligibility) == 38
assert len(values) == 74 and len(means) == 37 and len(points) == 148 and len(bars) == 16
assert len(asl_groups) == 16
assert order.Participant.tolist() == manifest["participant_order"]
assert order.Group.tolist() == ["Control"]*11+["ASD"]*8


def sigmoid(x, upper, lower, x0, k):
    return lower+(upper-lower)/(1+np.exp(-k*(x-x0)))


crossings = 0
for condition, expected_n in [("RT", 30), ("SD", 16)]:
    path = HERE / f"simultaneous_{condition.lower()}_trials.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["input_sha256"][path.name]
    data = pd.read_csv(path, float_precision="round_trip")
    assert data.Condition.eq(condition).all() and data.Valid.eq(1).all()
    assert len(data) == expected_n*20*19
    if condition == "SD":
        assert data.Stim_duration_s.eq(.03).all()
    data["delta_key"] = data.Delta_dB.round(2)
    measured = data.groupby(["Participant", "ASL_cd", "delta_key"]).Choice.agg(["size", lambda x: x.eq(1).sum()])
    recorded = counts.loc[counts.Condition.eq(condition)].set_index(["Participant", "ASL_cd", "delta_key"]).sort_index()
    assert measured["size"].eq(expected_n).all()
    np.testing.assert_array_equal(recorded.trials, measured["size"])
    np.testing.assert_array_equal(recorded.right_trials, measured.iloc[:, 1])
    np.testing.assert_allclose(recorded.p_right, measured.iloc[:, 1]/measured["size"], atol=1e-14, rtol=0)
    for fit in fits.loc[fits.Condition.eq(condition)].itertuples(index=False):
        p = recorded.loc[(fit.Participant, fit.ASL_cd)].sort_index()
        x, y = p.index.to_numpy(dtype=float), p.p_right.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parameters, _ = curve_fit(sigmoid, x, y, p0=[y.max(), y.min(), np.median(x), .1],
                                     bounds=([0,0,-np.inf,0], [1,1,np.inf,np.inf]), maxfev=10000)
        np.testing.assert_allclose(parameters, [fit.upper, fit.lower, fit.x0, fit.k], atol=1e-6, rtol=1e-6)
        high, low = .75*fit.upper, .25*(1-fit.lower)
        defined = fit.lower < low < high < fit.upper
        assert (fit.jnd_status == "ok") == defined
        if defined:
            bracket = (fit.x0-100/fit.k, fit.x0+100/fit.k)
            lower_crossing = brentq(lambda x: sigmoid(x, fit.upper, fit.lower, fit.x0, fit.k)-low, *bracket)
            upper_crossing = brentq(lambda x: sigmoid(x, fit.upper, fit.lower, fit.x0, fit.k)-high, *bracket)
            np.testing.assert_allclose([fit.delta_low, fit.delta_high, fit.jnd],
                                       [lower_crossing, upper_crossing, (upper_crossing-lower_crossing)/2], atol=1e-10, rtol=0)
            crossings += 2
        else:
            assert condition == "SD" and fit.Participant == 52 and fit.ASL_cd == 20
            assert low < fit.lower and np.isnan(fit.jnd)
assert crossings == 150


# %% Paired eligibility, per-condition centering, and original RT agreement
for row in eligibility.itertuples(index=False):
    own = fits.loc[fits.Condition.eq(row.Condition)&fits.Participant.eq(row.Participant)]
    paired = bool(own.jnd_status.eq("ok").all() and own.jnd.gt(0).all())
    assert row.eligible == paired
    selected = values.loc[values.Condition.eq(row.Condition)&values.Participant.eq(row.Participant)]
    if not paired:
        assert row.Condition == "SD" and row.Participant == 52 and selected.empty
        continue
    assert len(selected) == 2
    center = statistics.fmean(own.jnd.tolist())
    np.testing.assert_allclose(selected.participant_mean, center, atol=1e-14, rtol=0)
    np.testing.assert_allclose(selected.signed_delta, selected.jnd-center, atol=1e-14, rtol=0)
    assert abs(selected.signed_delta.sum()) < 1e-14
assert eligibility.groupby("Condition").eligible.sum().to_dict() == {"RT": 19, "SD": 18}
reference = pd.read_csv(HERE / "simultaneous_rt_metric_deviations/values_and_deviations.csv")
reference = reference.loc[reference.metric.eq("jnd")].set_index(["Participant", "ASL_cd"]).sort_index()
rt_values = values.loc[values.Condition.eq("RT")].set_index(["Participant", "ASL_cd"]).sort_index()
for field, ref_field in [("jnd", "value"), ("participant_mean", "participant_mean"), ("signed_delta", "signed_delta")]:
    np.testing.assert_allclose(rt_values[field], reference[ref_field], atol=1e-14, rtol=0)
for table, grouped in [(summary, False), (groups, True)]:
    for row in table.itertuples(index=False):
        own_values = values.loc[values.Condition.eq(row.Condition)]
        own_means = means.loc[means.Condition.eq(row.Condition)]
        if grouped:
            own_values = own_values.loc[own_values.Group.eq(row.Group)]
            own_means = own_means.loc[own_means.Group.eq(row.Group)]
        sample = own_means.participant_mean.tolist() if row.level == "participant_means" else own_values.signed_delta.tolist()
        center, spread = statistics.fmean(sample), statistics.pstdev(sample)
        assert row.n == len(sample) and row.ddof == 0
        np.testing.assert_allclose([row.mean, row.population_sd, row.lower, row.upper],
                                   [center, spread, center-spread, center+spread], atol=1e-14, rtol=0)
for row in asl_groups.itertuples(index=False):
    own = values.loc[values.Condition.eq(row.Condition)&values.Group.eq(row.Group)&values.ASL_cd.eq(row.ASL_cd)]
    field = "jnd" if row.level == "participant_ASL_values" else "signed_delta"
    sample = own[field].tolist()
    center, spread = statistics.fmean(sample), statistics.pstdev(sample)
    assert row.n == len(sample) and row.ddof == 0
    assert row.n == (11 if row.Group == "Control" else (8 if row.Condition == "RT" else 7))
    np.testing.assert_allclose([row.mean, row.population_sd, row.lower, row.upper],
                               [center, spread, center-spread, center+spread], atol=1e-14, rtol=0)
for (condition, group), table in asl_groups.loc[asl_groups.level.eq("ASL_deviations")].groupby(["Condition", "Group"]):
    assert abs(table["mean"].sum()) < 1e-14
    np.testing.assert_allclose(table.population_sd.iloc[0], table.population_sd.iloc[1], atol=1e-14, rtol=0)


# %% Check every saved PDF data mark, group bar, text label, and panel range
with Image.open(OUTPUT / manifest["png"]) as image:
    assert image.size == (3600, 3240)
with fitz.open(OUTPUT / manifest["pdf"]) as pdf:
    assert len(pdf) == 1
    page = pdf[0]
    text = page.get_text()
    assert "Control (n=11)" in text and "ASD (n=8)" in text and "ASD (n=7)" in text
    assert text.count("N/A") >= 3
    paths = page.get_drawings()
    for point in points.itertuples(index=False):
        selected = values.loc[values.Condition.eq(point.Condition)&values.Participant.eq(point.Participant)&values.ASL_cd.eq(point.ASL_cd)].iloc[0]
        assert point.marker == {"Control": "o", "ASD": "*"}[point.Group]
        expected_x = int(order.loc[order.Participant.eq(point.Participant), "position"].iloc[0]) + (-.105 if point.ASL_cd == 20 else .105)
        np.testing.assert_allclose([point.x, point.y], [expected_x, selected.jnd if point.panel == "raw" else selected.signed_delta], atol=1e-14, rtol=0)
        rgb = tuple(int(point.color[i:i+2], 16)/255 for i in [1,3,5])
        candidates = [p for p in paths if p["fill"] is not None and np.allclose(p["fill"], rgb, atol=1e-6)]
        centers = np.array([[(p["rect"].x0+p["rect"].x1)/2, (p["rect"].y0+p["rect"].y1)/2] for p in candidates])
        tolerance = .55 if point.marker == "*" else .04
        assert (np.max(abs(centers-[point.pdf_x_pt, point.pdf_y_pt]), axis=1) < tolerance).sum() == 1
        panel = limits.loc[limits.Condition.eq(point.Condition)&limits.panel.eq(point.panel)].iloc[0]
        assert panel.ylim_low <= point.y <= panel.ylim_high
    for bar in bars.itertuples(index=False):
        color = "#1f77b4" if bar.Group == "Control" else "#d62728"
        rgb = tuple(int(color[i:i+2], 16)/255 for i in [1,3,5])
        candidates = [p for p in paths if p["fill"] is None and p["color"] is not None and
                      np.isclose(p["width"], 4) and np.allclose(p["color"], rgb, atol=1e-6)]
        matched = []
        for path in candidates:
            for item in path["items"]:
                if item[0] == "l":
                    segment = [item[1].x, min(item[1].y,item[2].y), max(item[1].y,item[2].y)]
                    if np.max(abs(np.array(segment)-[bar.pdf_x_pt, bar.pdf_upper_y_pt, bar.pdf_lower_y_pt])) < .04:
                        matched.append(path)
        assert len(matched) == 1
        assert bar.linestyle == ("solid" if bar.ASL_cd == 20 else "dotted")
        assert (matched[0]["dashes"].replace(" ", "") == "[]0") == (bar.ASL_cd == 20)
        stat = asl_groups.loc[asl_groups.Condition.eq(bar.Condition)&asl_groups.Group.eq(bar.Group)&
                              asl_groups.ASL_cd.eq(bar.ASL_cd)&asl_groups.level.eq("participant_ASL_values" if bar.panel == "raw" else "ASL_deviations")].iloc[0]
        np.testing.assert_allclose([bar.n, bar.mean, bar.sd, bar.lower, bar.upper],
                                   [stat.n, stat["mean"], stat.population_sd, stat.lower, stat.upper], atol=1e-14, rtol=0)
        panel = limits.loc[limits.Condition.eq(bar.Condition)&limits.panel.eq(bar.panel)].iloc[0]
        assert panel.ylim_low <= bar.lower <= bar.upper <= panel.ylim_high
        assert bar.label in text
        assert f"ASL {bar.ASL_cd}" in bar.label
        for label in bar.label.splitlines()[2:]:
            found = [r for r in page.search_for(label) if abs((r.x0+r.x1)/2-bar.pdf_x_pt) < 1 and 0 < bar.pdf_upper_y_pt-r.y1 < 60]
            assert len(found) == 1
    for block in page.get_text("blocks"):
        assert block[0] >= 0 and block[1] >= 0 and block[2] <= page.rect.width and block[3] <= page.rect.height
result = {"status": "passed", "selected_trials": {"RT": 11400, "SD": 6080},
          "probabilities_checked": 760, "refits_checked": 76, "numeric_crossings_checked": crossings,
          "defined_jnds": 75, "complete_pairs": {"RT": 19, "SD": 18},
          "unavailable_pair": "SD Participant 52: ASL20 lower target is below the fitted lower asymptote",
          "rendered_data_marks_checked": 148, "group_asl_bars_and_labels_checked": 16,
          "group_asl_summaries_checked": 16, "bar_line_styles_checked": 16,
          "original_rt_values_match": True, "png_pixels": [3600, 3240]}
(OUTPUT / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
