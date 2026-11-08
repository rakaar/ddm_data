# %% Independent saved-draw, trial-population and figure audit
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_gaussian_delay_model_comparison_omega25'
SOURCES={'condition_gamma_omega':ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25',
         'npl_alpha':ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'}
ABLS=[20,40,60]
EXPECTED={'wt':(5,68184),'hom':(6,67715)}
manifest=json.loads((OUT/'manifest.json').read_text())
summary=pd.read_csv(OUT/'delay_posterior_comparison.csv',float_precision='round_trip')
differences=pd.read_csv(OUT/'delay_posterior_mean_differences.csv',float_precision='round_trip')
assert len(summary)==24 and len(differences)==12 and manifest['panels']==4
assert not summary.duplicated(['model','group','block','ABL']).any()
for relative,digest in manifest['source_hashes'].items():
    with (ROOT/relative).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,relative

for group,(animals,trials) in EXPECTED.items():
    folders=[p/group for p in SOURCES.values()]
    # No selection changes: identical compressed trial tables and fixed values.
    assert (folders[0]/'fit_trials.csv.gz').read_bytes()==(folders[1]/'fit_trials.csv.gz').read_bytes()
    assert json.loads((folders[0]/'fixed_proactive.json').read_text())['values']==json.loads((folders[1]/'fixed_proactive.json').read_text())['values']
    frame=pd.read_csv(folders[0]/'fit_trials.csv.gz',usecols=['animal','session_type','training_level','timed_rt','success'])
    assert len(frame)==trials and frame.animal.nunique()==animals
    assert frame.session_type.eq(1).all() and frame.training_level.eq(16).all()
    assert frame.success.isin([1,-1]).all() and frame.timed_rt.between(0,1).all()
    for model,folder in SOURCES.items():
        with np.load(folder/group/'seed_0/posterior_samples.npz') as saved:
            for block in ['delay_mu','delay_sigma']:
                assert saved[block].shape==(10000,3)
                for i,abl in enumerate(ABLS):
                    draws=saved[block][:,i]
                    row=summary.loc[summary.model.eq(model)&summary.group.eq(group)&summary.block.eq(block)&summary.ABL.eq(abl)].iloc[0]
                    expected=[np.mean(draws),np.percentile(draws,2.5),np.percentile(draws,97.5)]
                    np.testing.assert_allclose([row.mean_s,row.lower95_s,row.upper95_s],expected,atol=1e-14)
                    np.testing.assert_allclose([row.mean_ms,row.lower95_ms,row.upper95_ms],np.asarray(expected)*1000,atol=1e-11)
                    assert row.draws==10000 and row.animals==animals and row.valid_trials==trials
for row in differences.itertuples():
    z=summary.loc[summary.group.eq(row.group)&summary.block.eq(row.block)&summary.ABL.eq(row.ABL)].set_index('model')
    np.testing.assert_allclose(row.direct_minus_npl_mean_ms,z.loc['condition_gamma_omega','mean_ms']-z.loc['npl_alpha','mean_ms'],atol=1e-12)
shown=[]
for geometry in manifest['geometry']:
    np.testing.assert_allclose([geometry['width_inches'],geometry['height_inches']],[2.8,2.8],atol=1e-12)
    group,block=geometry['panel'].split('_',1)
    for artist in [a for a in manifest['artists'] if a['panel']==geometry['panel']]:
        assert artist['color']==manifest['colors'][artist['model']]
        assert artist['marker']==manifest['markers'][artist['model']]
        z=summary.loc[summary.group.eq(group)&summary.block.eq(block)&summary.model.eq(artist['model'])].sort_values('ABL')
        for key in ['mean_ms','lower95_ms','upper95_ms']: np.testing.assert_allclose(artist[key],z[key],atol=1e-12)
        assert artist['ABL']==ABLS
        assert min(artist['lower95_ms'])>geometry['ylim'][0] and max(artist['upper95_ms'])<geometry['ylim'][1]
        shown.extend((group,block,artist['model'],a) for a in ABLS)
assert len(shown)==24 and len(set(shown))==24
for block in ['delay_mu','delay_sigma']:
    scales=[g['ylim'] for g in manifest['geometry'] if g['panel'].endswith(block)]
    np.testing.assert_array_equal(scales[0],scales[1])
for name,digest in manifest['artifacts'].items():
    with (OUT/name).open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
    if name.endswith('.png'):
        with Image.open(OUT/name) as image: assert image.size==(1760,1760)
    if name.endswith('.pdf'): assert len(PdfReader(OUT/name).pages)==1
(OUT/'validation.json').write_text(json.dumps(dict(status='passed',panels=4,posterior_estimates=24,credible_intervals=24,
    posterior_draws_per_fit=10000,means_quantiles_units_verified=True,exact_trial_populations_match=True,
    fixed_proactive_lapse_values_match=True,source_hashes_unchanged=len(manifest['source_hashes']),
    equal_square_axes_verified=True,shared_column_scales=True,all_intervals_unclipped=True,
    artifacts=manifest['artifacts']),indent=2)+'\n')
print('Verified all 24 posterior estimates, CIs, source hashes, populations and four panels.')
