# %% Imports and parameters
"""Independently check saved fits, equal-weight means, signed deltas, and PDF dots.

Run after plot_session_type_1_animal_metric_deviations.py using .venv/bin/python.
"""
from collections import defaultdict
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

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_1_animal_metric_deviations"
manifest = json.loads((OUTPUT / "validation.json").read_text())
REFERENCE = REPO / manifest["reference_folder"]
ABLS = [20, 40, 60]
assert manifest["status"] == "passed" and manifest["rt_restriction"] is None
assert manifest["refitted"] is False and manifest["summary_SD_ddof"] == 0


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
assert len(eligibility) == 53 and len(means) == 103 and len(points) == 309 and len(summary) == 4
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
    assert len(m) == {"jnd":50,"k":53}[metric] and len(p) == {"jnd":150,"k":159}[metric]
    assert m.choice_trials.sum() == {"jnd":444718,"k":471416}[metric]
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
assert set(source)-set(eligible['jnd']) == {'ASD0035','ASD0044','ASD0048'}
assert set(eligible['k']) == set(source)
print('Independent arithmetic: 50 JND animals / 53 slope animals; 309 values and deviations',flush=True)


# %% Inspect actual PDF dot locations, summary bars, labels, colors, and full limits
plotted = pd.read_csv(OUTPUT / 'plotted_points.csv')
bars = pd.read_csv(OUTPUT / 'plotted_summary_bars.csv')
assert len(plotted) == 618 and len(bars) == 4
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
    'eligible_animals':{'jnd':50,'k':53},'eligible_choices':{'jnd':444718,'k':471416},
    'source_values_checked':309,'signed_deviations_checked':309,'saved_PDF_dots_checked':618,
    'animal_means_checked':103,'summary_bars_checked':4,'population_SD_ddof':0,
    'figure_checks':reports,'protected_files_checked':len(manifest['protected_file_sha256']),
    'checks':['independent source CSV parsing of saved fits','metric-specific complete eligibility',
        'equal ABL means and animal weighting','signed deviations sum to zero per animal',
        'independent ascending order and animal-ID tie break','population SDs from statistics.pstdev',
        'all PDF data dots and SD bar endpoints verified','full ranges and readable labels','unchanged input and previous artifacts']}
(OUTPUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Independent verification passed: 618 PDF dots, four population-SD bars, two PNG/PDF pairs, preserved earlier artifacts.')
