# %% Imports and parameters
"""Independently check saved fits, equal-weight means, signed deltas, and PDF dots.

Run after plot_session_type_1_animal_metric_deviations_calibrated.py using .venv/bin/python.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq, curve_fit

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_animal_metric_deviations_calibrated"
manifest = json.loads((OUTPUT / "validation.json").read_text())
REFERENCE = OUTPUT
ABLS = [20, 40, 60]
assert manifest["status"] == "passed" and manifest["rt_restriction"] is None
assert manifest["refitted"] is True and manifest["summary_SD_ddof"] == 0


# %% Independently reconstruct choices from both current and archived CSVs
cleanup = json.loads((REPO / "asd_analysis/session_type_csvs/speaker_calibration_cleanup.json").read_text())
lineage = cleanup["files"][manifest["input_csv"]]
condition_sets = {}
removed_by_animal = Counter()
for version, relative in [("before", lineage["archive_csv"]), ("after", manifest["input_csv"])]:
    cells = defaultdict(Counter)
    with (REPO / relative).open(newline="") as stream:
        for row in csv.DictReader(stream):
            assert float(row["training_level"]) == 16 and float(row["session_type"]) == 1
            assert row["trial_is_repeat"].lower() == "false"
            if version == "after":
                assert not (row["filename"] == "CNTNAP2_cohort4.csv" and float(row["session"]) < 13)
            success, abl, ild = float(row["success"] or "nan"), float(row["ABL"] or "nan"), float(row["ILD"] or "nan")
            abl = 60 if abl == 59 else abl
            if success not in [1,-1] or abl not in ABLS or abs(ild) not in [1,2,4,8,16]:
                continue
            assert float(row["response_poke"]) in [-1,1]
            key = (row["animal"], int(abl), int(ild))
            cells[key]["trials"] += 1
            cells[key]["right_trials"] += float(row["response_poke"]) == 1
            cells[key]["left_trials"] += float(row["response_poke"]) == -1
            cells[key]["correct_trials"] += success == 1
            if version == "before" and row["filename"] == "CNTNAP2_cohort4.csv" and float(row["session"]) < 13:
                removed_by_animal[row["animal"]] += 1
    condition_sets[version] = cells
    assert len(cells) == 1590
    assert sum(x["trials"] for x in cells.values()) == {"before":471416,"after":451381}[version]
assert sum(removed_by_animal.values()) == 20035 and len(removed_by_animal) == 10
with (OUTPUT / "slope_jnd_condition_counts.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1590
for row in rows:
    key = (row["animal"], int(float(row["plot_ABL"])), int(float(row["ILD"])))
    cell = condition_sets["after"][key]
    for field in ["trials", "right_trials", "left_trials", "correct_trials"]:
        assert int(row[field]) == cell[field]
    assert math.isclose(float(row["p_right"]), cell["right_trials"]/cell["trials"], rel_tol=0, abs_tol=2e-16)
with (OUTPUT / "slope_jnd_fits.csv").open(newline="") as stream:
    current_fits = {(r["animal"], int(r["plot_ABL"])):r for r in csv.DictReader(stream)}
old_reference = REPO / manifest["reference_folder"]
with (old_reference / "slope_jnd_fits.csv").open(newline="") as stream:
    previous_fits = {(r["group"].split("/")[-1], int(r["plot_ABL"])):r for r in csv.DictReader(stream) if r["view"] == "animal"}
assert len(current_fits) == len(previous_fits) == 159
fit_unchanged = 0
crossings = 0
for (animal, abl), row in current_fits.items():
    ilds = np.array([-16,-8,-4,-2,-1,1,2,4,8,16], dtype=float)
    cells = [condition_sets["after"][(animal,abl,int(i))] for i in ilds]
    measured = np.array([c["right_trials"]/c["trials"] for c in cells])
    # Refit the independently reconstructed ten points, with the original formula.
    def curve(x, upper, lower, x0, k):
        return lower + (upper-lower)/(1+np.exp(-k*(x-x0)))
    parameters, _ = curve_fit(curve, ilds, measured,
        p0=[max(measured),min(measured),float(np.median(ilds)),.1],
        bounds=([0,0,-np.inf,0],[1,1,np.inf,np.inf]), maxfev=10000)
    saved = np.array([float(row[k]) for k in ["upper","lower","x0","k"]])
    np.testing.assert_allclose(saved,parameters,rtol=0,atol=1e-10)
    upper, lower, x0, k = saved
    assert upper > lower and k > 0 and float(row["slope_k"]) == k
    assert int(row["trials"]) == sum(c["trials"] for c in cells)
    assert int(row["right_trials"]) == sum(c["right_trials"] for c in cells)
    high, low = .75*upper, .25*(1-lower)
    np.testing.assert_allclose([float(row["y_high"]),float(row["y_low"])],[high,low],rtol=0,atol=1e-15)
    if lower < low < high < upper:
        roots = [brentq(lambda x:curve(x,*saved)-target,x0-100/k,x0+100/k,xtol=1e-12) for target in [high,low]]
        np.testing.assert_allclose([float(row["ild_high"]),float(row["ild_low"]),float(row["jnd"])],
            [*roots,(roots[0]-roots[1])/2],rtol=0,atol=5e-11)
        crossings += 2
        assert row["jnd_status"] == "ok"
    else:
        assert row["jnd"] == "" and row["jnd_status"] == "thresholds_outside_asymptotes_or_reversed"
    old = previous_fits[(animal,abl)]
    if animal not in removed_by_animal:
        np.testing.assert_allclose(saved,[float(old[k]) for k in ["upper","lower","x0","k"]],rtol=0,atol=1e-10)
        fit_unchanged += 1
    for cell in cells:
        assert cell["trials"] > 0
assert fit_unchanged == 129 and crossings == 308
print("Independent raw selection, 159 refits, 154 JNDs and 129 unchanged conditions checked",flush=True)


# %% Reconstruct metric values/eligibility from source fits with the CSV parser
source = defaultdict(dict)
with (REFERENCE / "slope_jnd_fits.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if row["view"] != "animal":
            continue
        filename, animal = row["group"].split("/")
        assert int(row["plot_ABL"]) not in source[animal]
        source[animal][int(row["plot_ABL"])] = row
assert len(source) == 53 and all(set(rows) == set(ABLS) for rows in source.values())
eligibility = pd.read_csv(OUTPUT / "animal_eligibility.csv", keep_default_na=False)
means = pd.read_csv(OUTPUT / "animal_means_order.csv")
points = pd.read_csv(OUTPUT / "animal_abl_deviations.csv")
summary = pd.read_csv(OUTPUT / "summary_statistics.csv")
assert len(eligibility) == 53 and len(means) == 102 and len(points) == 306 and len(summary) == 4
expected_means, expected_deltas, eligible = {}, {}, {}
for metric, field in [("jnd", "jnd"), ("k", "slope_k")]:
    eligible[metric] = []
    for animal, rows in source.items():
        values = [float(rows[abl][field]) if rows[abl][field] else math.nan for abl in ABLS]
        complete = all(math.isfinite(x) and x>0 for x in values)
        audit = eligibility.loc[eligibility.animal.eq(animal)].iloc[0]
        assert bool(audit[f"{metric}_eligible"]) == complete
        assert audit.choice_trials == sum(int(r['trials']) for r in rows.values())
        assert audit.right_trials == sum(int(r['right_trials']) for r in rows.values())
        assert audit.left_trials == sum(int(r['left_trials']) for r in rows.values())
        for abl,value in zip(ABLS,values):
            actual = audit[f"{metric}_ABL_{abl}"]
            if math.isfinite(value):
                assert math.isclose(float(actual),value,rel_tol=0,abs_tol=2e-15)
            else:
                assert actual == ""
        if complete:
            eligible[metric].append(animal)
            center = statistics.fmean(values)
            expected_means[(metric,animal)] = center
            for abl,value in zip(ABLS,values):
                expected_deltas[(metric,animal,abl)] = value-center
    expected_order = sorted(eligible[metric],key=lambda animal:(expected_means[(metric,animal)],animal))
    m = means.loc[means.metric.eq(metric)]
    p = points.loc[points.metric.eq(metric)]
    assert m.animal.to_list() == expected_order
    assert m.position.to_list() == list(range(len(m))) and m.order_rank.to_list() == list(range(1,len(m)+1))
    assert len(m) == {"jnd":49,"k":53}[metric] and len(p) == {"jnd":147,"k":159}[metric]
    assert m.choice_trials.sum() == {"jnd":420616,"k":451381}[metric]
    for row in m.itertuples(index=False):
        assert math.isclose(row.animal_mean,expected_means[(metric,row.animal)],rel_tol=0,abs_tol=2e-14)
        assert row.choice_trials == sum(int(r['trials']) for r in source[row.animal].values())
        assert row.right_trials == sum(int(r['right_trials']) for r in source[row.animal].values())
        assert row.left_trials == sum(int(r['left_trials']) for r in source[row.animal].values())
        filename, _ = source[row.animal][20]['group'].split('/')
        assert row.filename == filename and row.genotype == source[row.animal][20]['genotype']
    for row in p.itertuples(index=False):
        saved = source[row.animal][row.plot_ABL]
        assert math.isclose(row.value,float(saved[field]),rel_tol=0,abs_tol=2e-14)
        assert math.isclose(row.animal_mean,expected_means[(metric,row.animal)],rel_tol=0,abs_tol=2e-14)
        assert math.isclose(row.signed_delta,expected_deltas[(metric,row.animal,row.plot_ABL)],rel_tol=0,abs_tol=2e-14)
        assert row.position == expected_order.index(row.animal)
        assert [row.trials,row.right_trials,row.left_trials] == [int(saved[c]) for c in ['trials','right_trials','left_trials']]
    np.testing.assert_allclose(p.groupby('animal').signed_delta.sum(),0,rtol=0,atol=3e-14)
    for level, values in [("animal_means",[expected_means[(metric,a)] for a in expected_order]),
                          ("ABL_deviations",[expected_deltas[(metric,a,b)] for a in expected_order for b in ABLS])]:
        row = summary.loc[summary.metric.eq(metric) & summary.level.eq(level)].iloc[0]
        center,sd = statistics.fmean(values),statistics.pstdev(values)
        assert row['n'] == len(values) and row.ddof == 0
        np.testing.assert_allclose([row['mean'],row.population_sd,row.lower,row.upper],
                                  [center,sd,center-sd,center+sd],rtol=0,atol=2e-14)
assert set(source)-set(eligible['jnd']) == {'ASD0035','ASD0044','ASD0048','ASD0058'}
assert set(eligible['k']) == set(source)
print('Independent arithmetic: 49 JND animals / 53 slope animals; 306 values and deviations',flush=True)


# %% Independently verify before/after count, fit, signed-delta and summary audits
condition_comparison = pd.read_csv(OUTPUT / "condition_count_comparison.csv")
for r in condition_comparison.itertuples(index=False):
    key = (r.animal,int(r.plot_ABL),int(r.ILD))
    for field in ["trials","right_trials","left_trials"]:
        before, after = condition_sets["before"][key][field], condition_sets["after"][key][field]
        assert getattr(r,field+"_before") == before and getattr(r,field+"_after") == after
        assert getattr(r,field+"_removed") == before-after
fit_comparison = pd.read_csv(OUTPUT / "fit_comparison.csv")
assert len(fit_comparison) == 159
for r in fit_comparison.itertuples(index=False):
    key = (r.animal,int(r.plot_ABL))
    before, after = previous_fits[key], current_fits[key]
    assert r.choices_removed == int(before["trials"])-int(after["trials"])
    for field in ["upper","lower","x0","k","jnd"]:
        for suffix, table in [("before",before),("after",after)]:
            actual = getattr(r,field+"_"+suffix)
            if table[field]:
                assert math.isclose(actual,float(table[field]),rel_tol=0,abs_tol=2e-14)
            else:
                assert math.isnan(actual)
        change = getattr(r,field+"_change")
        if before[field] and after[field]:
            assert math.isclose(change,float(after[field])-float(before[field]),rel_tol=0,abs_tol=2e-14)
        else:
            assert math.isnan(change)
summary_comparison = pd.read_csv(OUTPUT / "summary_comparison.csv")
assert len(summary_comparison) == 8
for r in summary_comparison.itertuples(index=False):
    field = "jnd" if r.metric == "jnd" else "slope_k"
    tables = {}
    for version, data in [("before",previous_fits),("after",current_fits)]:
        tables[version] = {a:[float(data[(a,b)][field]) for b in ABLS] for a in source
            if all(data[(a,b)][field] and float(data[(a,b)][field])>0 for b in ABLS)}
    if r.population == "same_animals_before_after":
        common = set(tables["before"]) & set(tables["after"])
        tables = {v:{a:x for a,x in t.items() if a in common} for v,t in tables.items()}
    calculated = {}
    for version, table in tables.items():
        assert getattr(r,"animals_"+version) == len(table)
        values = [statistics.fmean(x) for x in table.values()] if r.level == "animal_means" else [x-statistics.fmean(triple) for triple in table.values() for x in triple]
        center, sd = statistics.fmean(values), statistics.pstdev(values)
        np.testing.assert_allclose([getattr(r,"mean_"+version),getattr(r,"population_SD_"+version)],[center,sd],rtol=0,atol=2e-14)
        calculated[version] = (center,sd)
    assert math.isclose(r.SD_change_percent,100*(calculated["after"][1]/calculated["before"][1]-1),rel_tol=0,abs_tol=2e-12)
    assert math.isclose(r.mean_change,calculated["after"][0]-calculated["before"][0],rel_tol=0,abs_tol=2e-14)
comp = json.loads((OUTPUT/"comparison.json").read_text())
assert set(comp["changed_animals"]) == set(removed_by_animal)
assert comp["choices_removed"] == 20035 and comp["newly_ineligible_jnd"] == ["ASD0058"]


# %% Inspect actual PDF dot locations, summary bars, labels, colors, and full limits
plotted = pd.read_csv(OUTPUT / 'plotted_points.csv')
bars = pd.read_csv(OUTPUT / 'plotted_summary_bars.csv')
assert len(plotted) == 612 and len(bars) == 4
reports=[]
for figure in manifest['figures']:
    metric=figure['metric']
    data=points.loc[points.metric.eq(metric)]
    with Image.open(REPO/figure['png']) as png:
        assert png.size == (4680,1800)
        png.verify()
    with fitz.open(REPO/figure['pdf']) as document:
        assert len(document) == 1
        page=document[0]
        assert np.isclose(page.rect.width,26*72) and np.isclose(page.rect.height,10*72)
        text=page.get_text()
        assert set(re.findall(r'\bASD\d{4}\b',text)) == set(eligible[metric])
        assert 'Signed deviation:' in text and 'Mean ±1 SD' in text
        paths=page.get_drawings()
        for abl,color in manifest['colors'].items():
            abl=int(abl);rgb=tuple(int(color[i:i+2],16)/255 for i in [1,3,5])
            circles=[d for d in paths if d['fill'] is not None and np.allclose(d['fill'],rgb,atol=1e-6)
                     and any(item[0]=='c' for item in d['items'])]
            assert len(circles) == 2*len(eligible[metric])+1  # data in both panels plus page legend
            centers=np.array([[(d['rect'].x0+d['rect'].x1)/2,(d['rect'].y0+d['rect'].y1)/2] for d in circles])
            drawn=plotted.loc[plotted.metric.eq(metric) & plotted.plot_ABL.eq(abl)]
            for row in drawn.itertuples(index=False):
                value=data.loc[data.animal.eq(row.animal) & data.plot_ABL.eq(abl)].iloc[0]
                assert math.isclose(row.value,value['value' if row.panel=='raw' else 'signed_delta'],rel_tol=0,abs_tol=2e-14)
                assert row.color == color
                assert (np.max(abs(centers-[row.pdf_x_pt,row.pdf_y_pt]),axis=1)<.02).sum() == 1
        gray=[d for d in paths if d['fill'] is None and d['color'] is not None
              # PDF DeviceGray-to-RGB conversion can shift gray by about 1e-4.
              and np.allclose(d['color'],[128/255]*3,atol=2e-4) and np.isclose(d['width'],4)]
        assert len(gray)==2
        actual=[]
        for d in gray:
            for item in d['items']:
                if item[0]=='l':actual.append([item[1].x,min(item[1].y,item[2].y),max(item[1].y,item[2].y)])
        for bar in bars.loc[bars.metric.eq(metric)].itertuples(index=False):
            target=[bar.pdf_x_pt,bar.pdf_y_top_pt,bar.pdf_y_bottom_pt]
            assert (np.max(abs(np.array(actual)-target),axis=1)<.02).any()
            level='animal_means' if bar.panel=='raw' else 'ABL_deviations'
            s=summary.loc[summary.metric.eq(metric) & summary.level.eq(level)].iloc[0]
            np.testing.assert_allclose([bar.mean,bar.sd,bar.lower,bar.upper],[s['mean'],s.population_sd,s.lower,s.upper],rtol=0,atol=2e-14)
        assert figure['raw_ylim'][0]<=data.value.min() and figure['raw_ylim'][1]>=data.value.max()
        assert figure['delta_ylim'][0]<=data.signed_delta.min() and figure['delta_ylim'][1]>=data.signed_delta.max()
        assert np.isclose(-figure['delta_ylim'][0],figure['delta_ylim'][1])
        for level,limits in [('animal_means',figure['raw_ylim']),('ABL_deviations',figure['delta_ylim'])]:
            s=summary.loc[summary.metric.eq(metric) & summary.level.eq(level)].iloc[0]
            assert limits[0]<=s.lower and limits[1]>=s.upper
        for block in page.get_text('blocks'):
            assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width and block[3]<=page.rect.height
    reports.append({'metric':metric,'animals':len(eligible[metric]),'data_dots_checked':2*len(data),
                    'gray_SD_bars_checked':2,'pdf_pages':1,'png_pixels':[4680,1800]})


# %% Preserve inputs and all earlier analyses
for relative,digest in manifest['protected_file_sha256'].items():
    with (REPO/relative).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,relative
report={'status':'passed','source_script':str(Path(__file__).relative_to(REPO)),
    'eligible_animals':{'jnd':49,'k':53},'eligible_choices':{'jnd':420616,'k':451381},
    'source_values_checked':306,'signed_deviations_checked':306,'saved_PDF_dots_checked':612,
    'animal_means_checked':102,'summary_bars_checked':4,'population_SD_ddof':0,
    'figure_checks':reports,'protected_files_checked':len(manifest['protected_file_sha256']),
    'refits_checked':159,'numeric_JND_crossings_checked':308,'unchanged_animal_ABL_conditions':129,
    'before_after_summary_rows_checked':8,'checks':['independent current and archived CSV choice counts','all 159 refits and numeric JND crossings','eight common/all-eligible before-after summary rows','independent source CSV parsing of saved fits','metric-specific complete eligibility',
        'equal ABL means and animal weighting','signed deviations sum to zero per animal',
        'independent ascending order and animal-ID tie break','population SDs from statistics.pstdev',
        'all PDF data dots and SD bar endpoints verified','full ranges and readable labels','unchanged input and previous artifacts']}
(OUTPUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Independent verification passed: 612 PDF dots, four population-SD bars, two PNG/PDF pairs, preserved earlier artifacts.')
