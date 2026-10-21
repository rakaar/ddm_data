# %% Imports and parameters
"""Independently verify pooled raw trials, statistics, quantile scaling, and PDF."""
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_family_genotype_psychometric_rtd_chronometric_scaling"
manifest = json.loads((OUTPUT / "validation.json").read_text())
ABLS = [20, 40, 60]
ILDS = [1, 2, 4, 8, 16]
SIGNED = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ORDER = [f"{f}/{g}" for f in ["SHANK3", "CNTNAP2"] for g in ["wt", "het", "hom"]]
assert manifest["status"] == "passed" and manifest["group_order"] == ORDER


def sigmoid(x, upper, lower, x0, k):
    return lower + (upper-lower)/(1+np.exp(-k*(x-x0)))


# %% Parse actual CSV rows independently, without averaging animal summaries
raw, all_finite = defaultdict(list), defaultdict(list)
signed_choices, signed_right = Counter(), Counter()
condition_choices, positive_choices, positive_window = Counter(), Counter(), Counter()
member_choice, member_rt, member_window = Counter(), Counter(), Counter()
animal_identity = {}
input_rows = 0
with (REPO / manifest["input"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        input_rows += 1
        assert row["training_level"] == "16" and row["session_type"] == "1"
        assert row["trial_is_repeat"].lower() == "false" and float(row["is_short_sound"] or 0) == 0
        assert not (row["filename"] == "CNTNAP2_cohort4.csv" and float(row["session"]) < 13)
        abl = float(row["ABL"] or "nan")
        abl = 60 if abl == 59 else abl
        ild = float(row["ILD"] or "nan")
        if row["success"] not in {"1", "-1"} or abl not in ABLS or ild not in SIGNED:
            continue
        group_id = row["filename"].split("_")[0] + "/" + row["genotype"]
        assert group_id in ORDER and float(row["response_poke"]) in [-1, 1]
        animal = row["animal"]
        identity = (group_id, row["filename"], row["genotype"])
        if animal in animal_identity:
            assert animal_identity[animal] == identity
        animal_identity[animal] = identity
        signed = (group_id, int(abl), int(ild))
        key = (group_id, int(abl), int(abs(ild)))
        signed_choices[signed] += 1
        signed_right[signed] += float(row["response_poke"]) == 1
        condition_choices[key] += 1
        positive_choices[key] += ild > 0
        member_choice[(group_id, animal)] += 1
        rt = float(row["timed_rt"] or "nan")
        if not math.isfinite(rt):
            continue
        all_finite[key].append(rt)
        member_rt[(group_id, animal)] += 1
        if 0 <= rt <= 1:
            raw[key].append(rt)
            positive_window[key] += ild > 0
            member_window[(group_id, animal)] += 1
assert input_rows == 610603 and len(animal_identity) == 53 and len(signed_choices) == 180
assert sum(signed_choices.values()) == 451381 and sum(signed_right.values()) == 231263
assert sum(map(len, all_finite.values())) == 449193 and sum(map(len, raw.values())) == 446949
assert len(raw) == len(all_finite) == 90
membership = pd.read_csv(OUTPUT / "group_membership.csv")
assert len(membership) == 53
for r in membership.itertuples(index=False):
    assert animal_identity[r.animal] == (r.group_id, r.filename, r.genotype)
    assert [r.choice_trials, r.finite_rt_trials, r.window_trials] == [
        member_choice[(r.group_id, r.animal)], member_rt[(r.group_id, r.animal)], member_window[(r.group_id, r.animal)]]
groups = pd.read_csv(OUTPUT / "group_counts.csv")
assert groups.group_id.tolist() == ORDER and groups.animals.tolist() == [6, 7, 7, 10, 11, 12]
for r in groups.itertuples(index=False):
    assert r.choice_trials == sum(n for (g, a, i), n in signed_choices.items() if g == r.group_id)
    assert r.right_trials == sum(n for (g, a, i), n in signed_right.items() if g == r.group_id)
    assert r.finite_rt_trials == sum(len(v) for (g, a, i), v in all_finite.items() if g == r.group_id)
    assert r.window_trials == sum(len(v) for (g, a, i), v in raw.items() if g == r.group_id)
counts = pd.read_csv(OUTPUT / "condition_counts.csv")
assert len(counts) == 90
for r in counts.itertuples(index=False):
    key = (r.group_id, r.plot_ABL, r.abs_ILD)
    assert [r.valid_choice_trials, r.finite_rt_trials, r.window_trials] == [condition_choices[key], len(all_finite[key]), len(raw[key])]
    assert r.positive_ild_trials == positive_choices[key]
    assert r.negative_ild_trials == condition_choices[key] - positive_choices[key]
    assert r.positive_ild_window_trials == positive_window[key]
    assert r.negative_ild_window_trials == len(raw[key]) - positive_window[key]
    assert r.rt_above_1_s == sum(x > 1 for x in all_finite[key])


# %% Pooled probabilities and independent unweighted sigmoid refits
psy = pd.read_csv(OUTPUT / "psychometric_points.csv")
psy_fits = pd.read_csv(OUTPUT / "psychometric_fits.csv")
assert len(psy) == 180 and len(psy_fits) == 18
for r in psy.itertuples(index=False):
    key = (r.group_id, r.plot_ABL, r.ILD)
    assert r.trials == signed_choices[key] and r.right_trials == signed_right[key]
    assert math.isclose(r.p_right, signed_right[key]/signed_choices[key], rel_tol=0, abs_tol=2e-16)
for r in psy_fits.itertuples(index=False):
    x = np.array(SIGNED, dtype=float)
    p = np.array([signed_right[(r.group_id, r.plot_ABL, int(i))]/signed_choices[(r.group_id, r.plot_ABL, int(i))] for i in x])
    parameters, _ = curve_fit(sigmoid, x, p, p0=[max(p), min(p), float(np.median(x)), .1],
        bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]), maxfev=10000)
    np.testing.assert_allclose([r.upper, r.lower, r.x0, r.k], parameters, rtol=0, atol=1e-10)
    assert r.fit_status == "ok" and r.upper > r.lower and r.k > 0
    points = psy.loc[psy.group_id.eq(r.group_id) & psy.plot_ABL.eq(r.plot_ABL)].sort_values("ILD")
    np.testing.assert_allclose(points.fitted_p_right, sigmoid(x, *parameters), rtol=0, atol=2e-14)


# %% Full-range RTD bins and independent pooled means, sample SDs, and SEMs
histograms = pd.read_csv(OUTPUT / "rtd_histogram_bins.csv")
assert histograms.bin_count.sum() == 449193
for (group_id, abl), b in histograms.groupby(["group_id", "plot_ABL"]):
    b = b.sort_values("bin_index")
    edges = np.arange(len(b)+1)*.02
    values = [rt for ild in ILDS for rt in all_finite[(group_id, abl, ild)]]
    hist = [0]*len(b)
    for rt in values:
        hist[min(len(b)-1, bisect_right(edges, rt)-1)] += 1
    assert b.bin_count.tolist() == hist and b.normalization_trials.eq(len(values)).all()
    np.testing.assert_allclose(b.bin_left_s, edges[:-1], rtol=0, atol=2e-15)
    np.testing.assert_allclose(b.bin_right_s, edges[1:], rtol=0, atol=2e-15)
    np.testing.assert_allclose(b.density_per_s, np.array(hist)/(len(values)*.02), rtol=0, atol=1e-13)
    assert math.isclose(sum(b.density_per_s*.02), 1, rel_tol=0, abs_tol=2e-14)
chrono = pd.read_csv(OUTPUT / "chronometric_points.csv")
assert len(chrono) == 90 and chrono.finite_rt_trials.sum() == 449193
for r in chrono.itertuples(index=False):
    values = all_finite[(r.group_id, r.plot_ABL, r.abs_ILD)]
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    sem = sd/math.sqrt(len(values))
    np.testing.assert_allclose([r.mean_rt_s, r.sample_sd_rt_s, r.sem_rt_s, r.mean_minus_sem_s, r.mean_plus_sem_s],
        [mean, sd, sem, mean-sem, mean+sem], rtol=0, atol=2e-12)
    assert r.finite_rt_trials == len(values)
    assert math.isclose(r.plot_x, ILDS.index(r.abs_ILD)+{20: -.1, 40: 0, 60: .1}[r.plot_ABL], rel_tol=0, abs_tol=2e-15)
    assert 0 <= r.mean_minus_sem_s <= r.mean_plus_sem_s <= .45
for (group_id, abl), g in chrono.groupby(["group_id", "plot_ABL"]):
    values = [rt for ild in ILDS for rt in all_finite[(group_id, abl, ild)]]
    assert math.isclose(sum(g.mean_rt_s*g.finite_rt_trials)/g.finite_rt_trials.sum(), statistics.fmean(values), rel_tol=0, abs_tol=2e-12)
print("Checked raw pooling, 180 probabilities, 18 fits, all RTD bins and 90 means/SEMs", flush=True)


# %% Verify histogram bins and normalized CDFs using independent bisect counting
bins = pd.read_csv(OUTPUT / "histogram_cdf_bins.csv.gz")
assert len(bins) == 90000
cdfs, centers = {}, None
for (group_id, abl, ild), b in bins.groupby(["group_id", "plot_ABL", "abs_ILD"], sort=False):
    b = b.sort_values("bin_index")
    assert b.bin_index.to_list() == list(range(1000))
    edges = b.bin_left_s.to_list() + [float(b.bin_right_s.iloc[-1])]
    np.testing.assert_allclose(np.diff(edges), .001, rtol=0, atol=3e-16)
    assert edges[0] == 0 and edges[-1] == 1
    # Use the exact mathematical grid rather than CSV float-parser rounding at boundaries.
    edges = [float(x) for x in np.arange(0, 1.001, .001)]
    h = [0] * 1000
    for rt in raw[(group_id, abl, ild)]:
        h[min(999, bisect_right(edges, rt) - 1)] += 1
    assert b.hist_count.to_list() == h
    expected = [n / len(raw[(group_id, abl, ild)]) for n in accumulate(h)]
    np.testing.assert_allclose(b.cdf, expected, rtol=0, atol=3e-16)
    assert expected[-1] == 1 and np.all(np.diff(expected) >= 0)
    cdfs[(group_id, int(abl), int(ild))] = np.array(expected)
    if centers is None:
        centers = np.array([(a+b)/2 for a,b in zip(edges[:-1],edges[1:])])
    np.testing.assert_allclose(b.bin_center_s, centers, rtol=0, atol=3e-16)
assert bins.hist_count.sum() == 446949
del bins


# %% Verify every ΔCDF, both window means, and the first qualifying onset
delta = pd.read_csv(OUTPUT / "delta_cdf_curves.csv.gz")
thresholds = pd.read_csv(OUTPUT / "thresholds.csv").set_index(["group_id", "abs_ILD"])
assert len(delta) == 30000 and len(thresholds) == 30
independent_cutoffs = {}
for (group_id, ild), d in delta.groupby(["group_id", "abs_ILD"], sort=False):
    d = d.sort_values("bin_index")
    c20, c40, c60 = [cdfs[(group_id, abl, ild)] for abl in ABLS]
    pairwise = [abs(c20-c40), abs(c40-c60), abs(c20-c60)]
    expected = (pairwise[0]+pairwise[1]+pairwise[2])/3
    for field, values in zip(["abs_20_minus_40", "abs_40_minus_60", "abs_20_minus_60"], pairwise):
        np.testing.assert_allclose(d[field], values, rtol=0, atol=3e-16)
    np.testing.assert_allclose(d.mean_abs_delta_cdf, expected, rtol=0, atol=3e-16)
    np.testing.assert_allclose(expected, 2/3 * (np.maximum.reduce([c20,c40,c60])-np.minimum.reduce([c20,c40,c60])), rtol=0, atol=3e-16)
    # Scalar means on direct slices reproduce the paper's search, independently
    # of the plotting script's vectorized sliding-window implementation.
    previous = np.array([sum(expected[i-10:i])/10 for i in range(10,990)])
    following = np.array([sum(expected[i+1:i+11])/10 for i in range(10,990)])
    np.testing.assert_allclose(d.previous_mean.iloc[10:990], previous, rtol=0, atol=3e-16)
    np.testing.assert_allclose(d.following_mean.iloc[10:990], following, rtol=0, atol=3e-16)
    # Floating-point equality at exactly 0.005 is adjudicated with the original
    # paper's np.nanmean slices, not with a changed detection threshold.
    onset = next((i for i in range(10,990)
        if np.nanmean(expected[i+1:i+11])-np.nanmean(expected[i-10:i]) >= .005), None)
    r = thresholds.loc[(group_id,ild)]
    assert onset is not None and r.threshold_status == "ok" and r.onset_bin_index == onset
    assert math.isclose(r.cutoff_s, centers[onset], rel_tol=0, abs_tol=3e-16)
    assert math.isclose(r.cutoff_ms, centers[onset]*1000, rel_tol=0, abs_tol=3e-13)
    assert r.window_bins == 10 and r.required_rise == .005
    assert math.isclose(r.observed_rise, following[onset-10]-previous[onset-10], rel_tol=0, abs_tol=3e-16)
    independent_cutoffs[(group_id,int(ild))] = centers[onset]
assert math.isclose(thresholds.cutoff_ms.min(),37.5,abs_tol=1e-12) and math.isclose(thresholds.cutoff_ms.max(),79.5,abs_tol=1e-12)
del delta


# %% Independently calculate linear quantiles from sorted order statistics
def linear_quantile(sorted_values, probability):
    position = (len(sorted_values)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    weight = position-lo
    return sorted_values[lo] + weight*(sorted_values[hi]-sorted_values[lo])


sorted_rt = {key: sorted(values) for key,values in raw.items()}
raw_q = pd.read_csv(OUTPUT / "raw_quantiles.csv")
fine = pd.read_csv(OUTPUT / "fine_quantiles.csv.gz")
assert len(raw_q) == 450 and len(fine) == 8910
for table, value_field in [(raw_q,"raw_rt_s"), (fine,"rt_s")]:
    for r in table.itertuples(index=False):
        key = (r.group_id, r.plot_ABL, r.abs_ILD)
        expected = linear_quantile(sorted_rt[key], r.quantile)
        assert math.isclose(getattr(r,value_field),expected,rel_tol=0,abs_tol=2e-14)
fine_lookup = {(a,int(b),int(i)): q.sort_values("quantile") for (a,b,i),q in fine.groupby(["group_id","plot_ABL","abs_ILD"])}
fits = pd.read_csv(OUTPUT / "scaling_fits.csv")
fit_inputs = pd.read_csv(OUTPUT / "scaling_fit_inputs.csv.gz")
scaled = pd.read_csv(OUTPUT / "scaled_quantiles.csv")
assert len(fits) == 60 and len(fit_inputs) == 5940 and len(scaled) == 450
factors = {}
for r in fits.itertuples(index=False):
    t0 = independent_cutoffs[(r.group_id,r.abs_ILD)]
    # Recompute quantiles from the RTs rather than using rounded audit values.
    probabilities = np.arange(.01,1,.01)
    ref = [linear_quantile(sorted_rt[(r.group_id,60,r.abs_ILD)],q) for q in probabilities]
    other = [linear_quantile(sorted_rt[(r.group_id,r.plot_ABL,r.abs_ILD)],q) for q in probabilities]
    mask = [t0 <= x <= .3 for x in ref]
    x = [q-t0 for q,used in zip(ref,mask) if used]
    difference = [q-r0 for q,r0,used in zip(other,ref,mask) if used]
    intercept = difference[0]
    denominator = math.fsum(v*v for v in x)
    slope = math.fsum(xx*(yy-intercept) for xx,yy in zip(x,difference))/denominator
    factor = 1+slope
    assert r.fit_status == "ok" and factor > 0 and r.fit_points == sum(mask)
    np.testing.assert_allclose([r.cutoff_s,r.slope_m,r.scale_factor,r.first_difference_s,r.sum_x_squared],
                              [t0,slope,factor,intercept,denominator],rtol=0,atol=2e-12)
    inputs = fit_inputs.loc[fit_inputs.group_id.eq(r.group_id) & fit_inputs.plot_ABL.eq(r.plot_ABL) & fit_inputs.abs_ILD.eq(r.abs_ILD)].sort_values("quantile")
    assert inputs.used_for_fit.to_list() == mask
    np.testing.assert_allclose(inputs.reference_rt_s, ref, rtol=0, atol=2e-14)
    np.testing.assert_allclose(inputs.other_rt_s, other, rtol=0, atol=2e-14)
    np.testing.assert_allclose(inputs.shifted_difference_s,np.array(other)-ref-intercept,rtol=0,atol=2e-14)
    factors[(r.group_id,r.plot_ABL,r.abs_ILD)] = factor
for r in scaled.itertuples(index=False):
    expected_raw = linear_quantile(sorted_rt[(r.group_id,r.plot_ABL,r.abs_ILD)],r.quantile)
    if r.plot_ABL == 60:
        expected = expected_raw
        assert r.scaling_status == "reference_unchanged"
    else:
        t0 = independent_cutoffs[(r.group_id,r.abs_ILD)]
        expected = t0+(expected_raw-t0)/factors[(r.group_id,r.plot_ABL,r.abs_ILD)] if expected_raw>t0 else expected_raw
        assert r.scaling_status == "ok"
    np.testing.assert_allclose([r.raw_rt_s,r.scaled_rt_s],[expected_raw,expected],rtol=0,atol=2e-12)
print("Checked all 30 cutoffs, 9,360 raw/fine quantiles, 60 fits and 450 scaled points", flush=True)


# %% Exercise missing-onset and invalid-factor cases without executing the script
tree = ast.parse((REPO / manifest["source_script"]).read_text())
definitions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"detect_onset","fit_quantile_scale","rescale_quantiles"}]
namespace = {"np":np,"ONSET_WINDOW_BINS":10,"ONSET_RISE":.005,"FIT_MAX_RT":.3}
exec(compile(ast.Module(body=definitions,type_ignores=[]),"scaling_helpers","exec"),namespace)
assert namespace["detect_onset"](np.zeros(1000))[0] is None
assert namespace["detect_onset"](np.zeros(10))[0] is None
ref = np.array([.04,.08,.12,.2,.3])
result,_ = namespace["fit_quantile_scale"](ref,ref+.5*(ref-.04),.04)
assert result["fit_status"] == "ok" and np.isclose(result["scale_factor"],1.5)
np.testing.assert_allclose(namespace["rescale_quantiles"](ref+.5*(ref-.04),.04,1.5),ref)
assert namespace["fit_quantile_scale"](ref,ref,np.nan)[0]["fit_status"] == "missing_cutoff"
assert namespace["fit_quantile_scale"](ref,ref,.4)[0]["fit_status"] == "insufficient_fit_points"
assert namespace["fit_quantile_scale"](ref,-2*ref,.04)[0]["fit_status"] == "invalid_scale_factor"
assert np.isnan(namespace["rescale_quantiles"](ref,.04,-1)).all()
assert np.isnan(namespace["rescale_quantiles"](ref,np.nan,1)).all()


# %% Verify saved pages, every cutoff marker, existing means/SEMs, and unclipped panels
layout = pd.read_csv(OUTPUT / "panel_layout.csv")
plotted = pd.read_csv(OUTPUT / "plotted_thresholds.csv")
assert len(layout) == 6*9 and len(plotted) == 30
old_toc = [[1,f"{g.split('/')[0]} | {g.split('/')[1].upper()}",1+i//3] for i,g in enumerate(ORDER)]
psy_drawn = pd.read_csv(OUTPUT / "plotted_psychometric_points.csv")
assert len(psy_drawn) == 180
reports = []
with fitz.open(REPO / manifest["pdf"]) as pdf:
    assert len(pdf) == 2 and pdf.get_toc() == old_toc
    for record,page in zip(manifest["pages"],pdf):
        assert np.isclose(page.rect.width,18.24*72) and np.isclose(page.rect.height,record["page_height_in"]*72)
        assert len(record["group_ids"]) == 3
        text = page.get_text()
        assert record["group_ids"] == ORDER[3*(record["page"]-1):3*record["page"]]
        assert sum(line.strip().endswith("| Chronometric") for line in text.splitlines()) == len(record["group_ids"])
        # The threshold title contains all five ILDs, in milliseconds, for every group_id.
        for group_id in record["group_ids"]:
            for ild in ILDS:
                label = f"{ild}: {thresholds.loc[(group_id,ild),'cutoff_ms']:.1f}"
                assert label in text
        paths = page.get_drawings()
        for point in psy_drawn.loc[psy_drawn.pdf_page.eq(record["page"])].itertuples(index=False):
            p = psy.loc[psy.group_id.eq(point.group_id) & psy.plot_ABL.eq(point.plot_ABL) & psy.ILD.eq(point.ILD)].iloc[0]
            box = layout.loc[layout.group_id.eq(point.group_id) & layout.row_kind.eq("original") & layout.column.eq(1)].iloc[0]
            x = (box.x_in+(point.ILD-box.xlim_low)/(box.xlim_high-box.xlim_low)*box.width_in)*72
            y = (record["page_height_in"]-box.y_in-p.p_right*box.height_in)*72
            np.testing.assert_allclose([point.p_right,point.pdf_x_pt,point.pdf_y_pt],[p.p_right,x,y],rtol=0,atol=3e-11)
            rgb = tuple(int(point.color[i:i+2],16)/255 for i in [1,3,5])
            filled = [d for d in paths if d["fill"] is not None and np.allclose(d["fill"],rgb,atol=1e-6)]
            marker_centers = np.array([[(d["rect"].x0+d["rect"].x1)/2,(d["rect"].y0+d["rect"].y1)/2] for d in filled])
            assert (np.max(abs(marker_centers-[x,y]),axis=1)<.02).any()
        for group_id in record["group_ids"]:
            for ild in ILDS:
                r = plotted.loc[plotted.group_id.eq(group_id) & plotted.abs_ILD.eq(ild)].iloc[0]
                rgb = tuple(int(r.color[i:i+2],16)/255 for i in [1,3,5])
                found = []
                for drawing in paths:
                    if drawing["fill"] is None and drawing["color"] is not None and np.allclose(drawing["color"],rgb,atol=1e-6):
                        for item in drawing["items"]:
                            if item[0] == "l" and abs(item[1].x-item[2].x)<1e-4:
                                found.append([item[1].x,min(item[1].y,item[2].y),max(item[1].y,item[2].y)])
                target = [r.pdf_x_pt,r.pdf_y_top_pt,r.pdf_y_bottom_pt]
                assert len(found) and (np.max(abs(np.array(found)-target),axis=1)<.02).any(),(group_id,ild)
            group_id_layout = layout.loc[layout.group_id.eq(group_id)]
            c = group_id_layout.loc[group_id_layout.row_kind.eq("original") & group_id_layout.column.eq(3)].iloc[0]
            assert np.isclose(c.width_in,c.height_in)
            for abl in ABLS:
                rgb = tuple(int(manifest["ABL_colors"][str(abl)][i:i+2],16)/255 for i in [1,3,5])
                circles=[d for d in paths if d["fill"] is not None and np.allclose(d["fill"],rgb,atol=1e-6)
                         and any(item[0]=="c" for item in d["items"])]
                centers_pdf=np.array([[(d["rect"].x0+d["rect"].x1)/2,(d["rect"].y0+d["rect"].y1)/2] for d in circles])
                bars=[]
                for d in paths:
                    if d["fill"] is None and d["color"] is not None and np.allclose(d["color"],rgb,atol=1e-6):
                        bars.extend([[item[1].x,min(item[1].y,item[2].y),max(item[1].y,item[2].y)] for item in d["items"]
                                     if item[0]=="l" and abs(item[1].x-item[2].x)<1e-4])
                for point in chrono.loc[chrono.group_id.eq(group_id) & chrono.plot_ABL.eq(abl)].itertuples(index=False):
                    x=(c.x_in+(point.plot_x-c.xlim_low)/(c.xlim_high-c.xlim_low)*c.width_in)*72
                    y=(record["page_height_in"]-c.y_in-point.mean_rt_s/.45*c.height_in)*72
                    assert (np.max(abs(centers_pdf-[x,y]),axis=1)<.02).any()
                    top=(record["page_height_in"]-c.y_in-point.mean_plus_sem_s/.45*c.height_in)*72
                    bottom=(record["page_height_in"]-c.y_in-point.mean_minus_sem_s/.45*c.height_in)*72
                    assert (np.max(abs(np.array(bars)-[x,top,bottom]),axis=1)<.02).any()
            new = group_id_layout.loc[group_id_layout.row_kind.eq("new")]
            assert len(new) == 6
            assert new.loc[new.column.le(2),"xlim_high"].eq(.15).all()
            q = new.loc[new.column.ge(3)]
            assert q.ylim_high.nunique() == 1
            values = scaled.loc[scaled.group_id.eq(group_id),["raw_rt_s","scaled_rt_s"]].to_numpy()
            assert values.min()>=0 and values.max()<=q.ylim_high.iloc[0]
        for block in page.get_text("blocks"):
            assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width and block[3]<=page.rect.height
        with Image.open(REPO/record["png"]) as png:
            assert abs(png.width-18.24*160)<=1 and abs(png.height-record["page_height_in"]*160)<=1
            png.verify()
        reports.append({"page":record["page"],"group_ids":record["group_ids"],"cutoff_markers_checked":len(record["group_ids"])*5,
                        "chronometric_dots_and_SEMs_checked":len(record["group_ids"])*15})
for relative,digest in manifest["protected_file_sha256"].items():
    with (REPO/relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == digest,relative
report={"status":"passed","source_script":str(Path(__file__).relative_to(REPO)),
    "group_ids":6,"new_window_trials":446949,"histogram_cdf_conditions":90,"histogram_bins_checked":90000,
    "delta_cdf_bins_checked":30000,"thresholds_checked":30,"scaling_fits_checked":60,
    "raw_quantile_points_checked":450,"fine_quantiles_checked":8910,"scaled_points_checked":450,
    "synthetic_checks":["flat/short curve has no onset","known factor 1.5 recovered and inverted",
        "missing cutoff and insufficient fit points audited","nonpositive factor omitted rather than used"],
    "raw_pooling_and_18_refits_checked":True,"psychometric_probabilities_checked":180,"PDF_psychometric_dots_checked":180,"chronometric_dots_and_SEMs_checked_in_PDF":90,
    "pdf_pages":2,"genotype_bookmarks_checked":6,"page_checks":reports,
    "protected_files_checked":len(manifest["protected_file_sha256"])}
(OUTPUT/"independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print("Independent verification passed: raw counts, CDFs, 30 cutoffs, 60 fits, 2 PDF pages, preserved original data.")
