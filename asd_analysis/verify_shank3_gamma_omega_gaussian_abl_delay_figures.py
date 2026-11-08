# %% Reconstruct every displayed bin, point, interval and pooling weight
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from PIL import Image
from PyPDF2 import PdfReader
from shank3_gamma_omega_gaussian_abl_delay_utils import OUT, REFERENCES, COUNTS, GLOBALS, sha256, PRIORS

artifacts={}
for group in ['wt','hom']:
    folder=OUT/group
    n=COUNTS[group]
    assert json.loads((folder/'independent_validation.json').read_text())['status']=='passed'
    checked=json.loads((folder/'independent_validation.json').read_text())
    run=json.loads((folder/'seed_0/run_summary.json').read_text())
    np.testing.assert_allclose(run['posterior_mean_joint_log_likelihood'],checked['joint_log_likelihood'],atol=1e-7,rtol=1e-12)
    samples=dict(np.load(folder/'seed_0/posterior_samples.npz'))
    mapped=pd.read_csv(folder/'condition_parameter_means.csv',float_precision='round_trip')
    for block in ['delay_mu','delay_sigma']:
        np.testing.assert_allclose(mapped[block],np.repeat(samples[block].mean(0),10),atol=1e-12)
    valid=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    conditions=pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    bins=pd.read_csv(folder/'rtd_data_bins.csv',float_precision='round_trip')
    old=pd.read_csv(REFERENCES[group]/'rtd_data_bins.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(bins,old,check_exact=True)
    assert bins['count'].sum()==n['display']
    signed=pd.read_csv(folder/'signed_parent_model_1ms.csv.gz',float_precision='round_trip')
    pooled=pd.read_csv(folder/'pooled_rtd_model_1ms.csv.gz',float_precision='round_trip')
    mb=pd.read_csv(folder/'pooled_rtd_model_bins.csv',float_precision='round_trip')
    for (abl,ild),sub in pooled.groupby(['ABL','abs_ILD']):
        pairs=[signed.loc[signed.ABL.eq(abl)&signed.ILD.eq(s*ild)].sort_values('rt_s') for s in [-1,1]]
        weights=[x.parent_trials.iloc[0] for x in pairs]
        raw=np.average([x.unnormalized_density for x in pairs],axis=0,weights=weights)
        cdf=np.average([x.unconditional_cdf for x in pairs],axis=0,weights=weights)
        mass=cdf[-1]-cdf[0]
        np.testing.assert_allclose(raw/mass,sub.density,atol=1e-12)
        np.testing.assert_allclose((cdf-cdf[0])/mass,sub.window_cdf,atol=1e-12)
        np.testing.assert_allclose(sub.window_cdf.iloc[[0,-1]],[0,1],atol=1e-12)
        assert np.diff(sub.window_cdf).min()>-1e-7
        z=mb.loc[mb.ABL.eq(abl)&mb.abs_ILD.eq(ild)]
        np.testing.assert_allclose(z.window_mass,np.diff(cdf[::20])/mass,atol=1e-12)
        np.testing.assert_allclose(z.window_mass.sum(),1,atol=1e-12)
        assert z.window_mass.min()>-1e-7
    psych=pd.read_csv(folder/'psychometric_predictions.csv',float_precision='round_trip')
    assert len(psych)==30 and psych.trials.sum()==n['valid']
    assert psych.model_p_right.between(0,1).all()
    np.testing.assert_allclose(psych.integrated_both_choices,1,atol=2e-6)
    for row in psych.itertuples():
        z=valid.loc[valid.condition.eq(row.condition)]
        assert len(z)==row.trials and z.response_poke.eq(1).sum()==row.right
        np.testing.assert_allclose(row.data_p_right,z.response_poke.eq(1).mean(),atol=1e-15)
    manifest=json.loads((folder/'plot_manifest.json').read_text())
    assert manifest['display_xlim']==[-.1,.6] and manifest['rtd_bin_width_s']==.02
    assert manifest['model_linewidth_pt']==.9 and manifest['data_linewidth_pt']==1.4
    assert manifest['ABL_colors']=={'20':'#1f77b4','40':'#ff7f0e','60':'#2ca02c'}
    assert manifest['models'][0]['rtd_panels']==15
    for name,digest in manifest['artifacts'].items():
        assert sha256(folder/name)==digest
        with Image.open(folder/name) as im: assert im.width>1000 and im.height>1000
        artifacts[str((folder/name).relative_to(OUT))]=digest

estimates=pd.read_csv(OUT/'parameter_estimates.csv',float_precision='round_trip')
moments=pd.read_csv(OUT/'actual_truncated_delay_moments.csv',float_precision='round_trip')
assert len(estimates)==136
for group in ['wt','hom']:
    samples=dict(np.load(OUT/group/'seed_0/posterior_samples.npz'))
    for row in estimates.loc[estimates.group.eq(group)].itertuples():
        x=samples[row.block] if row.block in GLOBALS else samples[row.block][:,int(row.index)]
        np.testing.assert_allclose([row.mean,row.lower95,row.upper95],[x.mean(),*np.quantile(x,[.025,.975])],atol=1e-12)
        np.testing.assert_allclose([row.display_mean,row.display_lower95,row.display_upper95],np.array([row.mean,row.lower95,row.upper95])*row.display_scale,atol=1e-12)
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    m,var=truncnorm.stats(-mu/sd,np.inf,loc=mu,scale=sd,moments='mv')
    for row in moments.loc[moments.group.eq(group)].itertuples():
        x=m[:,row.abl_index] if row.quantity=='truncated_mean' else np.sqrt(var[:,row.abl_index])
        np.testing.assert_allclose([row.posterior_mean_s,row.lower95_s,row.upper95_s],[x.mean(),*np.quantile(x,[.025,.975])],atol=1e-11)
boundaries=pd.read_csv(OUT/'posterior_boundary_audit.csv',float_precision='round_trip')
assert len(boundaries)==136
for row in estimates.itertuples():
    with np.load(OUT/row.group/'seed_0/posterior_samples.npz') as saved:
        x=saved[row.block] if row.block in GLOBALS else saved[row.block][:,int(row.index)]
    bounds=np.asarray(PRIORS[row.block])
    lo,hi=(bounds[0,int(row.index)],bounds[-1,int(row.index)]) if bounds.ndim==2 else (bounds[0],bounds[-1])
    q=boundaries.loc[boundaries.group.eq(row.group)&boundaries.parameter.eq(row.parameter)].iloc[0]
    np.testing.assert_allclose([q.hard_lower,q.hard_upper,q.fraction_near_lower,q.fraction_near_upper],
        [lo,hi,np.mean(x<lo+.01*(hi-lo)),np.mean(x>hi-.01*(hi-lo))],atol=1e-14)
    assert np.all(x>lo) and np.all(x<hi)
ranges=pd.read_csv(OUT/'reference_evidence_parameter_ranges.csv')
for row in ranges.itertuples():
    old=pd.read_csv(REFERENCES[row.group]/'condition_parameter_means.csv')
    np.testing.assert_allclose([row.minimum,row.maximum],[old[row.parameter].min(),old[row.parameter].max()],atol=1e-12)
manifest=json.loads((OUT/'parameter_plot_manifest.json').read_text())
assert manifest['estimates']==136 and manifest['panels']==10
assert manifest['colors']=={'wt':'#000000','hom':'#d62728'}
shown=[]
for geometry in manifest['geometry']:
    np.testing.assert_allclose([geometry['width_inches'],geometry['height_inches']],[2.8,2.8],atol=1e-12)
    for artist in [x for x in manifest['artists'] if x['panel']==geometry['panel']]:
        assert min(artist['lower95'])>geometry['ylim'][0] and max(artist['upper95'])<geometry['ylim'][1]
        shown.extend((artist['group'],n) for n in artist['parameters'])
assert len(shown)==136 and len(set(shown))==136
assert set(shown)==set(zip(estimates.group,estimates.parameter))
for name,digest in manifest['artifacts'].items():
    assert sha256(OUT/name)==digest
    if name.endswith('.pdf'): assert len(PdfReader(OUT/name).pages)==1
    artifacts[name]=digest
(OUT/'figure_validation.json').write_text(json.dumps(dict(status='passed',rtd_panels=30,psychometric_points=60,
    parameter_estimates=136,parameter_panels=10,posterior_boundary_estimates=136,empirical_arrays_unchanged=True,
    signed_parent_weights_reconstructed=True,actual_truncated_moments_verified=True,
    artifacts=artifacts,verifier_sha256=sha256(Path(__file__))),indent=2)+'\n')
print('All numerical figure checks passed; 5 PNGs and a 1-page PDF.',flush=True)
