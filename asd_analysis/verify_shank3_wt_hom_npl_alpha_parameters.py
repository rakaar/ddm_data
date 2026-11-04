# %% Independent reconstruction from both saved draw arrays and condition tables
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_comparison'
GLOBALS = ['rate_lambda','T_0','theta_E','rate_norm_l','alpha','w','del_go']
ABLS,ILDS = [20,40,60],[-16,-8,-4,-2,-1,1,2,4,8,16]
manifest = json.loads((OUT/'plot_manifest.json').read_text())
estimates = pd.read_csv(OUT/'parameter_estimates.csv',float_precision='round_trip')
assert len(estimates)==74 and estimates.groupby('group').size().to_dict()=={'hom':37,'wt':37}
assert manifest['colors']=={'wt':'#000000','hom':'#d62728'}
assert manifest['panels']==10 and manifest['global_panels']==7 and manifest['delay_panels']==3
for group in ['wt','hom']:
    fit = ROOT/f'asd_analysis/shank3_cohort1_{group}_npl_alpha_condition_delay'
    draws = np.load(fit/'seed_0/posterior_samples.npz')
    conditions = pd.read_csv(fit/'conditions.csv')
    subset = estimates.loc[estimates.group.eq(group)]
    assert subset.parameter.is_unique
    for row in subset.itertuples():
        if row.block=='t_E_aff':
            condition = conditions.loc[conditions.condition.eq(int(row.condition))].iloc[0]
            assert row.ABL==condition.ABL and row.ILD==condition.ILD
            vector = draws['t_E_aff'][:,int(row.condition)]
            assert row.parameter==f't_E_aff_{int(row.condition):02d}'
        else:
            assert row.block in GLOBALS and row.parameter==row.block
            vector = draws[row.block]
        assert len(vector)==10000
        expected = np.array([np.sum(vector)/len(vector),*np.percentile(vector,[2.5,97.5])])
        np.testing.assert_allclose([row.mean,row.lower95,row.upper95],expected,atol=1e-13,rtol=1e-12)
        scale = 1000. if row.block in ['T_0','del_go','t_E_aff'] else 1.
        assert row.display_scale==scale
        np.testing.assert_allclose([row.display_mean,row.display_lower95,row.display_upper95],expected*scale,atol=1e-10)
assert len(manifest['artists'])==20
seen = []
for artist in manifest['artists']:
    group,panel = artist['group'],artist['panel']
    indexed = estimates.loc[estimates.group.eq(group)].set_index('parameter')
    values = indexed.loc[artist['parameters']]
    assert artist['color']==manifest['colors'][group]
    np.testing.assert_allclose(artist['means'],values.display_mean,atol=1e-10)
    np.testing.assert_allclose(artist['lower95'],values.display_lower95,atol=1e-10)
    np.testing.assert_allclose(artist['upper95'],values.display_upper95,atol=1e-10)
    if panel.startswith('delay_'):
        abl = int(panel.split('_')[1])
        assert values.ILD.tolist()==ILDS and values.ABL.eq(abl).all()
        offset = -.10 if group=='wt' else .10
        np.testing.assert_allclose(artist['x'],np.arange(10)+offset,atol=1e-14)
    else:
        assert artist['parameters']==[panel]
        np.testing.assert_array_equal(artist['x'],[0 if group=='wt' else 1])
    axis = next(g for g in manifest['geometry'] if g['panel']==panel)
    assert min(artist['lower95'])>axis['ylim'][0] and max(artist['upper95'])<axis['ylim'][1]
    seen.extend((group,p) for p in artist['parameters'])
assert len(seen)==len(set(seen))==74
for panel in manifest['geometry']:
    np.testing.assert_allclose([panel['width_inches'],panel['height_inches']],[2.8,2.8],atol=1e-12)
    if panel['panel'].startswith('delay_'):
        np.testing.assert_array_equal(panel['ylim'],manifest['delay_ylim'])
for relative,digest in json.loads((OUT/'input_preservation.json').read_text()).items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
for name,digest in manifest['artifacts'].items():
    path = OUT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    if path.suffix=='.pdf':
        info = subprocess.check_output(['pdfinfo',str(path)],text=True)
        assert int(next(x.split(':')[1] for x in info.splitlines() if x.startswith('Pages:')))==1
    else:
        with Image.open(path) as im:
            assert im.size==(3200,2720)
report = dict(status='passed',parameter_estimates=74,draws_per_genotype=10000,global_panels=7,delay_panels=3,
    condition_specific_delay_estimates=60,all_parameters_shown_once=True,all_panels_equal_square=True,
    delay_ILDS=ILDS,colors=manifest['colors'],all_errorbar_endpoints_verified=True,
    all_intervals_unclipped=True,inputs_and_fit_outputs_unchanged=True,pdf_pages=1,
    artifacts=manifest['artifacts'],verifier=str(Path(__file__).relative_to(ROOT)),
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified all 74 posterior estimates, 10 square panels, colors, mappings and input preservation.',flush=True)
