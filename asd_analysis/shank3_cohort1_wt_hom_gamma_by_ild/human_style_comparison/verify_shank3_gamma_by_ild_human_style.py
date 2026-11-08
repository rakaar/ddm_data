# %% Independent raw-trial pooling, posterior, artist and PDF checks
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import subprocess
import shutil
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from shank3_gamma_by_ild_utils import ROOT, OUT, ABLS, ILDS, EXPECTED, sha256

DEST = OUT/'human_style_comparison'
ABS_ILDS = [1,2,4,8,16]
manifest = json.loads((DEST/'plot_manifest.json').read_text())
hist = pd.read_csv(DEST/'pooled_rtd_histograms.csv',float_precision='round_trip')
curves = pd.read_csv(DEST/'pooled_rtd_model_curves.csv.gz',float_precision='round_trip')
psych = pd.read_csv(DEST/'pooled_psychometrics.csv',float_precision='round_trip')
chrono = pd.read_csv(DEST/'chronometric_points.csv',float_precision='round_trip')
parameters = pd.read_csv(DEST/'parameter_summary.csv',float_precision='round_trip')
weights = pd.read_csv(DEST/'psychometric_pooling_weights.csv',float_precision='round_trip')
signed_means = pd.read_csv(OUT/'signed_condition_means.csv',float_precision='round_trip')
with gzip.open(DEST/'plotted_series.json.gz','rt') as stream:traces=json.load(stream)
assert len(psych)==20 and len(chrono)==10 and len(parameters)==32 and len(weights)==60
assert not manifest['refitted']
max_mean_error = 0.

for group,e in EXPECTED.items():
    folder = OUT/group
    valid = pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    parent = pd.read_csv(folder/'parent_trials.csv.gz',float_precision='round_trip')
    predictions = pd.read_csv(folder/'gamma_by_ild/predictions/psychometric_predictions.csv',float_precision='round_trip')
    unnormalized = pd.read_csv(folder/'gamma_by_ild/predictions/signed_parent_model_1ms.csv.gz',float_precision='round_trip')
    conditions = pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    draws = dict(np.load(folder/'gamma_by_ild/posterior_samples.npz'))
    assert len(valid)==e['valid'] and valid.animal.nunique()==e['animals']
    assert valid.success.isin([1,-1]).all() and valid.timed_rt.between(0,1).all()
    assert valid.response_poke.eq(1).sum()==e['right'] and valid.response_poke.eq(-1).sum()==e['left']
    abort_rt = parent.fix_time/1000-parent.intended_fix_time/1000
    is_valid = parent.success.isin([1,-1]) & np.isfinite(parent.timed_rt) & parent.timed_rt.between(0,1)
    is_abort = parent.abort_type.eq('Fixation') & np.isfinite(abort_rt) & abort_rt.ge(-1) & abort_rt.lt(0)
    displayed = parent.loc[is_valid|is_abort].copy()
    displayed['display_rt'] = np.where(is_valid.loc[displayed.index],displayed.timed_rt,abort_rt.loc[displayed.index])
    assert len(displayed)==e['valid']+e['aborts']
    for ild in ABS_ILDS:
        h = hist.loc[hist.group.eq(group)&hist.abs_ILD.eq(ild)].sort_values('bin')
        m = curves.loc[curves.group.eq(group)&curves.abs_ILD.eq(ild)].sort_values('rt_s')
        edges = np.r_[h.left,h.right.iloc[-1]]
        raw = displayed.loc[displayed.ILD.abs().eq(ild)]
        np.testing.assert_array_equal(h['count'],np.histogram(raw.display_rt,edges)[0])
        np.testing.assert_allclose(h.density,h['count']/len(raw)/.02,atol=1e-12)
        np.testing.assert_allclose(np.sum(h.density*np.diff(edges)),1,atol=1e-12)
        np.testing.assert_allclose(np.diff(edges),.02,atol=1e-14)
        np.testing.assert_allclose(np.diff(m.rt_s),.001,atol=1e-14)
        pieces = conditions.loc[conditions.ILD.abs().eq(ild)]
        source = [unnormalized.loc[unnormalized.condition.eq(i)].sort_values('rt_s') for i in pieces.condition]
        q = np.average([x.unnormalized_density for x in source],axis=0,weights=pieces.parent_trials)
        f = np.average([x.unconditional_cdf for x in source],axis=0,weights=pieces.parent_trials)
        mass = f[-1]-f[0]
        np.testing.assert_allclose(m.model_density,q/mass,atol=1e-12)
        np.testing.assert_allclose(m.window_cdf,(f-f[0])/mass,atol=1e-12)
        np.testing.assert_allclose(trapezoid(m.model_density,m.rt_s),1,atol=3e-4)
        np.testing.assert_allclose(m.window_cdf.iloc[[0,-1]],[0,1],atol=1e-12)
        c = chrono.loc[chrono.group.eq(group)&chrono.abs_ILD.eq(ild)].iloc[0]
        raw_valid = valid.loc[valid.ILD.abs().eq(ild)]
        assert c.n==len(raw_valid)
        np.testing.assert_allclose(c.data_mean_RT_s,raw_valid.timed_rt.mean(),atol=1e-14)
        animal_rt = raw_valid.groupby('animal').timed_rt.mean()
        assert c.animals==len(animal_rt)
        np.testing.assert_allclose(c.animal_SEM_s,animal_rt.std(ddof=1)/np.sqrt(len(animal_rt)),atol=1e-14)
        signed = signed_means.loc[signed_means.group.eq(group)&signed_means.abs_ILD.eq(ild)]
        frequencies = [int(valid.condition.eq(i).sum()) for i in signed.condition]
        reference_mean = np.average(signed.model_mean_RT_s,weights=frequencies)
        np.testing.assert_allclose(c.model_mean_RT_s,reference_mean,atol=1e-12)
        np.testing.assert_allclose(c.model_mean_RT_s,c.model_mean_from_saved_CDF_s,atol=1e-6)
        max_mean_error = max(max_mean_error,abs(c.model_mean_RT_s-reference_mean))
    for ild in ILDS:
        row = psych.loc[psych.group.eq(group)&psych.ILD.eq(ild)].iloc[0]
        selected = valid.loc[valid.ILD.eq(ild)]
        assert row.trials==len(selected) and row.right==selected.response_poke.eq(1).sum()
        np.testing.assert_allclose(row.data_p_right,selected.response_poke.eq(1).mean(),atol=1e-14)
        animal_prob = selected.assign(right=selected.response_poke.eq(1)).groupby('animal').right.mean()
        assert row.animals==len(animal_prob)
        np.testing.assert_allclose([row.unweighted_animal_mean_p_right,row.animal_sample_SD,row.animal_SEM],
            [animal_prob.mean(),animal_prob.std(ddof=1),animal_prob.std(ddof=1)/np.sqrt(len(animal_prob))],atol=1e-14)
        components = predictions.loc[predictions.ILD.eq(ild)].sort_values('ABL')
        counts = np.array([valid.condition.eq(i).sum() for i in components.condition])
        reference_probability = np.average(components.model_p_right,weights=counts)
        np.testing.assert_allclose(row.model_p_right,reference_probability,atol=1e-14)
        w = weights.loc[weights.group.eq(group)&weights.ILD.eq(ild)].sort_values('ABL')
        np.testing.assert_allclose(w.weight,counts/len(selected),atol=1e-14)
    for row in parameters.loc[parameters.group.eq(group)].itertuples():
        index = int(row.parameter.rsplit('_',1)[1]) if row.block=='gamma' else ABLS.index(int(row.ABL)) if row.block=='omega' else 0
        values = draws[row.block].reshape(10000,-1)[:,index]
        actual = np.r_[values.mean(),np.quantile(values,[.025,.975])]
        np.testing.assert_allclose([row.mean,row.lower95,row.upper95],actual,atol=1e-12)
        scale = 1000 if row.block in ['t_E_aff','del_go'] else 1
        np.testing.assert_allclose([row.display_mean,row.display_lower95,row.display_upper95],actual*scale,atol=1e-12)
    print(group.upper()+': raw counts, all pooling weights, means and posterior intervals passed',flush=True)

# %% Check actual plotted series, colors, styles and axis geometry
colors = {'wt':'#4d4d4d','hom':'#c24a7a'}
for trace in traces:
    group,role = trace['group'],trace['role']
    assert trace['color']==colors[group]
    if role.startswith('rtd'):
        model = role=='rtd_model'
        table = (curves if model else hist)
        table = table.loc[table.group.eq(group)&table.abs_ILD.eq(trace['abs_ILD'])].sort_values('rt_s' if model else 'bin')
        np.testing.assert_array_equal(trace['y'],table.model_density if model else table.density)
        np.testing.assert_array_equal(trace['x'] if model else trace['edges'],table.rt_s if model else np.r_[table.left,table.right.iloc[-1]])
        assert trace['alpha']==(1 if model else .5)
        assert trace['linewidth']==(1.2 if model else 1.4)
    elif role.startswith('psychometric'):
        table = psych.loc[psych.group.eq(group)].sort_values('ILD')
        np.testing.assert_array_equal(trace['x'],table.ILD)
        np.testing.assert_array_equal(trace['y'],table.model_p_right if role.endswith('model') else table.data_p_right)
        if role.endswith('data'):
            assert trace['kind']=='errorbar'
            np.testing.assert_allclose([trace['lower'],trace['upper']],
                [table.data_p_right-table.animal_SEM,table.data_p_right+table.animal_SEM],atol=1e-14)
    elif role.startswith('chronometric'):
        table = chrono.loc[chrono.group.eq(group)].sort_values('abs_ILD')
        np.testing.assert_array_equal(trace['x'],table.abs_ILD)
        np.testing.assert_array_equal(trace['y'],table.model_mean_RT_s if role.endswith('model') else table.data_mean_RT_s)
        assert trace['kind']==('line' if role.endswith('model') else 'errorbar')
        if role.endswith('data'):
            np.testing.assert_allclose([trace['lower'],trace['upper']],
                [table.data_mean_RT_s-table.animal_SEM_s,table.data_mean_RT_s+table.animal_SEM_s],atol=1e-14)
    else:
        table = parameters.loc[parameters.group.eq(group)&parameters.block.eq(trace['block'])].sort_values('ILD' if trace['block']=='gamma' else 'ABL')
        np.testing.assert_allclose([trace['y'],trace['lower'],trace['upper']],table[['display_mean','display_lower95','display_upper95']].to_numpy().T,atol=1e-12)
        assert trace['kind']=='errorbar' and trace['linestyle']=='None'
    if role.endswith('model'):assert trace['linewidth']==1.2 and trace['linestyle']=='-'
assert len(manifest['figures'])==3
for figure in manifest['figures']:
    behavior = figure['kind'].startswith('behavior')
    expected_size = [26,6] if behavior else [14,12]
    assert figure['size_inches']==expected_size
    assert len(figure['panels'])==(7 if behavior else 5)
    for panel in figure['panels']:
        assert not panel['grid_visible']
        if panel['panel']=='del_go':assert panel['title']=='del_go' and panel['ylabel']=='ms'
        if panel['panel']=='gamma':np.testing.assert_allclose(panel['width_inches'],panel['height_inches'],atol=1e-12)
        if panel['panel'].startswith('rtd_'):assert panel['xlim']==([0,.6] if figure['kind']=='behavior_zoom' else [-.1,.6])
        for trace in [t for t in traces if t['figure']==('behavior' if behavior else 'parameters') and t['panel']==panel['panel']]:
            y = np.asarray(trace['y'])
            if 'lower' in trace:y=np.r_[y,trace['lower'],trace['upper']]
            assert y.min()>=panel['ylim'][0]-1e-12 and y.max()<=panel['ylim'][1]+1e-12
    info=subprocess.check_output(['pdfinfo',str(DEST/(figure['stem']+'.pdf'))],text=True)
    assert int(next(x.split(':')[1] for x in info.splitlines() if x.startswith('Pages:')))==1
for name,expected in manifest['artifacts'].items():assert sha256(DEST/name)==expected
for name,expected in manifest['source_hashes'].items():assert sha256(Path(name))==expected,name
assert sha256(Path(manifest['source']))==manifest['source_sha256']
report = dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),single_page_PDFs=3,
    behavior_panels=7,parameter_panels=5,pooled_psychometric_points=20,chronometric_pairs=10,
    parameter_means_and_intervals=32,rtd_model_curves=10,raw_counts_and_normalization_verified=True,
    requested_colors_verified=True,square_Gamma_verified=True,source_hashes_unchanged=True,
    grids_removed=True,readout_title_verified=True,RTD_display_ranges_verified=[[-.1,.6],[0,.6]],
    across_animal_SEMs_verified=30,
    max_chronometric_weighting_error_s=float(max_mean_error),visual_inspection='pending',
    verifier_source_sha256=sha256(Path(__file__)))
(DEST/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(Path(__file__),DEST/Path(__file__).name)
print(json.dumps(report,indent=2),flush=True)
