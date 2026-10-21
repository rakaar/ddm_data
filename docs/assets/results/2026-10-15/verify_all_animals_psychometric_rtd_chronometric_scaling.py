# %% Imports and parameters
"""Independent raw-CSV checks for the extended CDF/quantile-scaling PDF.

Use stdlib CSV, bisect histograms, explicit linear order-statistic quantiles,
and scalar least-squares arithmetic. Do not import the executing plot script.
"""
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import re
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric_scaling"
manifest = json.loads((OUTPUT / "validation.json").read_text())
REFERENCE = REPO / manifest["reference_output"]
ABLS, ILDS = [20, 40, 60], [1, 2, 4, 8, 16]
assert manifest["status"] == "passed" and manifest["repeat_filter_reapplied"] is False


# %% Reconstruct the complete valid population and new-window signed raw trials
raw, all_finite = defaultdict(list), defaultdict(list)
choices, positive, signed_choices, signed_right = Counter(), Counter(), Counter(), Counter()
identities, selected_animals = {}, set()
prepared_rows = full_choices = finite_count = 0
with (REPO / manifest["input"]).open(newline="") as stream:
    for row in csv.DictReader(stream):
        prepared_rows += 1
        assert float(row["session_type"]) == 1 and float(row["training_level"]) == 16
        assert row["trial_is_repeat"] == "False" and float(row["is_short_sound"]) == 0
        if float(row["success"]) not in [1, -1]:
            continue
        abl = int(float(row["ABL"]))
        abl = 60 if abl == 59 else abl
        ild = float(row["ILD"])
        if abl not in ABLS or abs(ild) not in ILDS:
            continue
        animal = row["animal"]
        identity = (row["filename"], row["genotype"])
        assert animal not in identities or identities[animal] == identity
        identities[animal] = identity
        selected_animals.add(animal)
        key = (animal, abl, int(abs(ild)))
        choices[key] += 1
        signed_choices[(animal, abl, int(ild))] += 1
        signed_right[(animal, abl, int(ild))] += float(row["response_poke"]) == 1
        full_choices += 1
        rt = float(row["timed_rt"] or "nan")
        if math.isfinite(rt):
            all_finite[key].append(rt)
            finite_count += 1
            if 0 <= rt <= 1:
                raw[key].append(rt)
                positive[key] += ild > 0
assert (prepared_rows, full_choices, finite_count, sum(map(len, raw.values()))) == (642370, 471416, 469228, 466978)
assert len(selected_animals) == 53 and len(raw) == 795
assert {"ASD0035", "ASD0044", "ASD0048", "ASD0026", "ASD0045"} <= selected_animals
counts = pd.read_csv(OUTPUT / "condition_counts.csv")
for r in counts.itertuples(index=False):
    key = (r.animal, r.plot_ABL, r.abs_ILD)
    assert (r.filename, r.genotype) == identities[r.animal]
    assert r.window_trials == len(raw[key]) and r.valid_choice_trials == choices[key]
    assert r.finite_rt_trials == len(all_finite[key])
    assert r.positive_ild_trials == positive[key] and r.negative_ild_trials == len(raw[key]) - positive[key]
    assert r.rt_above_1_s == sum(x > 1 for x in all_finite[key])
    assert r.nonfinite_rt_trials == choices[key] - len(all_finite[key])
assert counts.window_trials.sum() == 466978 and counts.window_trials.min() == 29
print("Independent selection: all 53 animals; 466,978 window RTs", flush=True)


# %% Verify histogram bins and normalized CDFs using independent bisect counting
bins = pd.read_csv(OUTPUT / "histogram_cdf_bins.csv.gz")
assert len(bins) == 795000
cdfs, centers = {}, None
for (animal, abl, ild), b in bins.groupby(["animal", "plot_ABL", "abs_ILD"], sort=False):
    b = b.sort_values("bin_index")
    assert b.bin_index.to_list() == list(range(1000))
    edges = b.bin_left_s.to_list() + [float(b.bin_right_s.iloc[-1])]
    np.testing.assert_allclose(np.diff(edges), .001, rtol=0, atol=3e-16)
    assert edges[0] == 0 and edges[-1] == 1
    # Use the exact mathematical grid rather than CSV float-parser rounding at boundaries.
    edges = [float(x) for x in np.arange(0, 1.001, .001)]
    h = [0] * 1000
    for rt in raw[(animal, abl, ild)]:
        h[min(999, bisect_right(edges, rt) - 1)] += 1
    assert b.hist_count.to_list() == h
    expected = [n / len(raw[(animal, abl, ild)]) for n in accumulate(h)]
    np.testing.assert_allclose(b.cdf, expected, rtol=0, atol=3e-16)
    assert expected[-1] == 1 and np.all(np.diff(expected) >= 0)
    cdfs[(animal, int(abl), int(ild))] = np.array(expected)
    if centers is None:
        centers = np.array([(a+b)/2 for a,b in zip(edges[:-1],edges[1:])])
    np.testing.assert_allclose(b.bin_center_s, centers, rtol=0, atol=3e-16)
assert bins.hist_count.sum() == 466978
del bins


# %% Verify every ΔCDF, both window means, and the first qualifying onset
delta = pd.read_csv(OUTPUT / "delta_cdf_curves.csv.gz")
thresholds = pd.read_csv(OUTPUT / "thresholds.csv").set_index(["animal", "abs_ILD"])
assert len(delta) == 265000 and len(thresholds) == 265
independent_cutoffs = {}
for (animal, ild), d in delta.groupby(["animal", "abs_ILD"], sort=False):
    d = d.sort_values("bin_index")
    c20, c40, c60 = [cdfs[(animal, abl, ild)] for abl in ABLS]
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
    r = thresholds.loc[(animal,ild)]
    assert onset is not None and r.threshold_status == "ok" and r.onset_bin_index == onset
    assert math.isclose(r.cutoff_s, centers[onset], rel_tol=0, abs_tol=3e-16)
    assert math.isclose(r.cutoff_ms, centers[onset]*1000, rel_tol=0, abs_tol=3e-13)
    assert r.window_bins == 10 and r.required_rise == .005
    assert math.isclose(r.observed_rise, following[onset-10]-previous[onset-10], rel_tol=0, abs_tol=3e-16)
    independent_cutoffs[(animal,int(ild))] = centers[onset]
assert math.isclose(thresholds.cutoff_ms.min(),10.5,abs_tol=1e-12) and math.isclose(thresholds.cutoff_ms.max(),104.5,abs_tol=1e-12)
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
assert len(raw_q) == 3975 and len(fine) == 78705
for table, value_field in [(raw_q,"raw_rt_s"), (fine,"rt_s")]:
    for r in table.itertuples(index=False):
        key = (r.animal, r.plot_ABL, r.abs_ILD)
        expected = linear_quantile(sorted_rt[key], r.quantile)
        assert math.isclose(getattr(r,value_field),expected,rel_tol=0,abs_tol=2e-14)
fine_lookup = {(a,int(b),int(i)): q.sort_values("quantile") for (a,b,i),q in fine.groupby(["animal","plot_ABL","abs_ILD"])}
fits = pd.read_csv(OUTPUT / "scaling_fits.csv")
fit_inputs = pd.read_csv(OUTPUT / "scaling_fit_inputs.csv.gz")
scaled = pd.read_csv(OUTPUT / "scaled_quantiles.csv")
assert len(fits) == 530 and len(fit_inputs) == 52470 and len(scaled) == 3975
factors = {}
for r in fits.itertuples(index=False):
    t0 = independent_cutoffs[(r.animal,r.abs_ILD)]
    # Recompute quantiles from the RTs rather than using rounded audit values.
    probabilities = np.arange(.01,1,.01)
    ref = [linear_quantile(sorted_rt[(r.animal,60,r.abs_ILD)],q) for q in probabilities]
    other = [linear_quantile(sorted_rt[(r.animal,r.plot_ABL,r.abs_ILD)],q) for q in probabilities]
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
    inputs = fit_inputs.loc[fit_inputs.animal.eq(r.animal) & fit_inputs.plot_ABL.eq(r.plot_ABL) & fit_inputs.abs_ILD.eq(r.abs_ILD)].sort_values("quantile")
    assert inputs.used_for_fit.to_list() == mask
    np.testing.assert_allclose(inputs.reference_rt_s, ref, rtol=0, atol=2e-14)
    np.testing.assert_allclose(inputs.other_rt_s, other, rtol=0, atol=2e-14)
    np.testing.assert_allclose(inputs.shifted_difference_s,np.array(other)-ref-intercept,rtol=0,atol=2e-14)
    factors[(r.animal,r.plot_ABL,r.abs_ILD)] = factor
for r in scaled.itertuples(index=False):
    expected_raw = linear_quantile(sorted_rt[(r.animal,r.plot_ABL,r.abs_ILD)],r.quantile)
    if r.plot_ABL == 60:
        expected = expected_raw
        assert r.scaling_status == "reference_unchanged"
    else:
        t0 = independent_cutoffs[(r.animal,r.abs_ILD)]
        expected = t0+(expected_raw-t0)/factors[(r.animal,r.plot_ABL,r.abs_ILD)] if expected_raw>t0 else expected_raw
        assert r.scaling_status == "ok"
    np.testing.assert_allclose([r.raw_rt_s,r.scaled_rt_s],[expected_raw,expected],rtol=0,atol=2e-12)
print("Checked all 265 cutoffs, 82,680 raw/fine quantiles, 530 fits and 3,975 scaled points", flush=True)


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


# %% Existing numerical audits remain byte-identical and agree with raw selection
for name in manifest["copied_reference_audits"]:
    assert (OUTPUT/f"reference_{name}").read_bytes() == (REFERENCE/name).read_bytes()
members = pd.read_csv(REFERENCE / "animal_pages.csv")
assert members.animal.to_list() == manifest["animal_order"]
psy = pd.read_csv(REFERENCE / "psychometric_points.csv")
for r in psy.itertuples(index=False):
    key = (r.animal,int(r.plot_ABL),int(r.ILD))
    assert r.trials == signed_choices[key] and r.right_trials == signed_right[key]
chrono = pd.read_csv(REFERENCE / "chronometric_points.csv")
for r in chrono.itertuples(index=False):
    values = all_finite[(r.animal,r.plot_ABL,r.abs_ILD)]
    np.testing.assert_allclose([r.mean_rt_s,r.sem_rt_s],
        [statistics.fmean(values),statistics.stdev(values)/math.sqrt(len(values))],rtol=0,atol=1e-12)


# %% Verify saved pages, every cutoff marker, existing means/SEMs, and unclipped panels
layout = pd.read_csv(OUTPUT / "panel_layout.csv")
plotted = pd.read_csv(OUTPUT / "plotted_thresholds.csv")
assert len(layout) == 53*9 and len(plotted) == 265
with fitz.open(REPO / json.loads((REFERENCE/"validation.json").read_text())["pdf"]) as old:
    old_toc = old.get_toc()
reports = []
with fitz.open(REPO / manifest["pdf"]) as pdf:
    assert len(pdf) == 11 and pdf.get_toc() == old_toc
    for record,page in zip(manifest["pages"],pdf):
        assert np.isclose(page.rect.width,18.24*72) and np.isclose(page.rect.height,record["page_height_in"]*72)
        assert len(record["animals"]) == (5 if record["page"]<11 else 3)
        text = page.get_text()
        assert set(re.findall(r"\bASD\d{4}\b",text)) == set(record["animals"])
        assert sum(line.strip().endswith("| Chronometric") for line in text.splitlines()) == len(record["animals"])
        # The threshold title contains all five ILDs, in milliseconds, for every animal.
        for animal in record["animals"]:
            for ild in ILDS:
                label = f"{ild}: {thresholds.loc[(animal,ild),'cutoff_ms']:.1f}"
                assert label in text
        paths = page.get_drawings()
        for animal in record["animals"]:
            for ild in ILDS:
                r = plotted.loc[plotted.animal.eq(animal) & plotted.abs_ILD.eq(ild)].iloc[0]
                rgb = tuple(int(r.color[i:i+2],16)/255 for i in [1,3,5])
                found = []
                for drawing in paths:
                    if drawing["fill"] is None and drawing["color"] is not None and np.allclose(drawing["color"],rgb,atol=1e-6):
                        for item in drawing["items"]:
                            if item[0] == "l" and abs(item[1].x-item[2].x)<1e-4:
                                found.append([item[1].x,min(item[1].y,item[2].y),max(item[1].y,item[2].y)])
                target = [r.pdf_x_pt,r.pdf_y_top_pt,r.pdf_y_bottom_pt]
                assert len(found) and (np.max(abs(np.array(found)-target),axis=1)<.02).any(),(animal,ild)
            animal_layout = layout.loc[layout.animal.eq(animal)]
            c = animal_layout.loc[animal_layout.row_kind.eq("original") & animal_layout.column.eq(3)].iloc[0]
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
                for point in chrono.loc[chrono.animal.eq(animal) & chrono.plot_ABL.eq(abl)].itertuples(index=False):
                    x=(c.x_in+(point.plot_x-c.xlim_low)/(c.xlim_high-c.xlim_low)*c.width_in)*72
                    y=(record["page_height_in"]-c.y_in-point.mean_rt_s/.45*c.height_in)*72
                    assert (np.max(abs(centers_pdf-[x,y]),axis=1)<.02).any()
                    top=(record["page_height_in"]-c.y_in-point.mean_plus_sem_s/.45*c.height_in)*72
                    bottom=(record["page_height_in"]-c.y_in-point.mean_minus_sem_s/.45*c.height_in)*72
                    assert (np.max(abs(np.array(bars)-[x,top,bottom]),axis=1)<.02).any()
            new = animal_layout.loc[animal_layout.row_kind.eq("new")]
            assert len(new) == 6
            assert new.loc[new.column.le(2),"xlim_high"].eq(.15).all()
            q = new.loc[new.column.ge(3)]
            assert q.ylim_high.nunique() == 1
            values = scaled.loc[scaled.animal.eq(animal),["raw_rt_s","scaled_rt_s"]].to_numpy()
            assert values.min()>=0 and values.max()<=q.ylim_high.iloc[0]
        for block in page.get_text("blocks"):
            assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width and block[3]<=page.rect.height
        with Image.open(REPO/record["png"]) as png:
            assert abs(png.width-18.24*160)<=1 and abs(png.height-record["page_height_in"]*160)<=1
            png.verify()
        reports.append({"page":record["page"],"animals":record["animals"],"cutoff_markers_checked":len(record["animals"])*5,
                        "chronometric_dots_and_SEMs_checked":len(record["animals"])*15})
for relative,digest in manifest["protected_file_sha256"].items():
    with (REPO/relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == digest,relative
report={"status":"passed","source_script":str(Path(__file__).relative_to(REPO)),
    "animals":53,"new_window_trials":466978,"histogram_cdf_conditions":795,"histogram_bins_checked":795000,
    "delta_cdf_bins_checked":265000,"thresholds_checked":265,"scaling_fits_checked":530,
    "raw_quantile_points_checked":3975,"fine_quantiles_checked":78705,"scaled_points_checked":3975,
    "synthetic_checks":["flat/short curve has no onset","known factor 1.5 recovered and inverted",
        "missing cutoff and insufficient fit points audited","nonpositive factor omitted rather than used"],
    "copied_reference_audits_byte_identical":True,"chronometric_dots_and_SEMs_checked_in_PDF":795,
    "pdf_pages":11,"original_bookmarks_preserved":True,"page_checks":reports,
    "protected_files_checked":len(manifest["protected_file_sha256"])}
(OUTPUT/"independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print("Independent verification passed: raw counts, CDFs, 265 cutoffs, 530 fits, 11 PDF pages, preserved original data.")
