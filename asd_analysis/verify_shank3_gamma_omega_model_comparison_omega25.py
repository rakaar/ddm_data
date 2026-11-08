# %% Independent posterior-interval and hyperbolic NPL+alpha mapping audit
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_model_comparison_omega25'
SOURCES={'condition_gamma_omega':ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25',
         'npl_alpha':ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'}
ABLS=[20,40,60]
ILDS=[-16,-8,-4,-2,-1,1,2,4,8,16]
PARAMETERS=['rate_lambda','T_0','theta_E','rate_norm_l','alpha']
m=json.loads((OUT/'manifest.json').read_text())
points=pd.read_csv(OUT/'direct_gamma_omega_posterior_points.csv',float_precision='round_trip')
curves=pd.read_csv(OUT/'npl_alpha_continuous_gamma_omega.csv',float_precision='round_trip')
parameters=pd.read_csv(OUT/'npl_alpha_mapping_parameter_means.csv',float_precision='round_trip')
assert len(points)==120 and len(curves)==1926 and len(parameters)==10
assert m['panels']==4 and m['plotted_curves']==8
for relative,digest in m['source_hashes'].items():
    with (ROOT/relative).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest,relative
with (ROOT/'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py').open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==m['formula_sha256']
max_error=0.
for group,n in [('wt',68184),('hom',67715)]:
    direct=SOURCES['condition_gamma_omega']/group
    npl=SOURCES['npl_alpha']/group
    assert (direct/'fit_trials.csv.gz').read_bytes()==(npl/'fit_trials.csv.gz').read_bytes()
    assert json.loads((direct/'fixed_proactive.json').read_text())['values']==json.loads((npl/'fixed_proactive.json').read_text())['values']
    conditions=pd.read_csv(direct/'conditions.csv')
    assert len(conditions)==30 and conditions.trials.sum()==n
    with np.load(direct/'seed_0/posterior_samples.npz') as saved:
        for row in points.loc[points.group.eq(group)].itertuples():
            assert saved[row.block].shape==(10000,30)
            x=saved[row.block][:,row.condition]
            np.testing.assert_allclose([row.mean,row.lower95,row.upper95],[x.mean(),np.percentile(x,2.5),np.percentile(x,97.5)],atol=1e-12)
            old=conditions.loc[conditions.condition.eq(row.condition)].iloc[0]
            assert old.ABL==row.ABL and old.ILD==row.ILD and old.trials==row.condition_trials
            expected_offset={20:-.18,40:0.,60:.18}[row.ABL]
            np.testing.assert_allclose(row.displayed_ILD,row.ILD+expected_offset,atol=1e-14)
    with np.load(npl/'seed_0/posterior_samples.npz') as saved:
        means={name:saved[name].mean() for name in PARAMETERS}
    reported=parameters.loc[parameters.group.eq(group)].set_index('parameter')
    np.testing.assert_allclose([reported.loc[name,'posterior_mean'] for name in PARAMETERS],[means[name] for name in PARAMETERS],atol=1e-14)
    lam,t0,theta,ell,alpha=[means[name] for name in PARAMETERS]
    for abl in ABLS:
        z=curves.loc[curves.group.eq(group)&curves.ABL.eq(abl)].sort_values('ILD')
        assert len(z)==321 and z.ILD.iloc[0]==-16 and z.ILD.iloc[-1]==16
        np.testing.assert_allclose(np.diff(z.ILD),.1,atol=6e-15,rtol=0)
        # Algebraically independent hyperbolic form of the two normalized rates.
        u=lam*z.ILD.to_numpy()/17.37
        even=np.cosh((1-ell)*u)+alpha*np.cosh((1+ell)*u)
        odd=np.sinh((1-ell)*u)+alpha*np.sinh((1+ell)*u)
        gamma=theta*odd/even
        omega=2*10**(lam*(1-ell)*abl/20)*even/((1+alpha*alpha+2*alpha*np.cosh(2*ell*u))*t0*theta*theta)
        for name,expected in [('gamma',gamma),('omega',omega)]:
            np.testing.assert_allclose(z[name],expected,atol=1e-12,rtol=1e-13)
            max_error=max(max_error,float(np.max(np.abs(z[name].to_numpy()-expected))))
        np.testing.assert_allclose(z.gamma,-z.gamma.to_numpy()[::-1],atol=1e-13)
        np.testing.assert_allclose(z.omega,z.omega.to_numpy()[::-1],atol=1e-12)
        np.testing.assert_allclose(z.gamma.to_numpy()[160],0.,atol=1e-14)
        reference=pd.read_csv(npl/'condition_parameter_means.csv',float_precision='round_trip')
        for ild in ILDS:
            i=int(round((ild+16)*10))
            old=reference.loc[reference.ABL.eq(abl)&reference.ILD.eq(ild)].iloc[0]
            np.testing.assert_allclose([z.gamma.iloc[i],z.omega.iloc[i]],[old.gamma,old.omega],atol=1e-12)
    gamma_by_abl=[curves.loc[curves.group.eq(group)&curves.ABL.eq(a)].sort_values('ILD').gamma.to_numpy() for a in ABLS]
    np.testing.assert_allclose(gamma_by_abl,np.broadcast_to(gamma_by_abl[0],(3,321)),atol=1e-13)
shown_points=[];shown_curves=[]
for geometry in m['geometry']:
    np.testing.assert_allclose([geometry['width_inches'],geometry['height_inches']],[3.4,3.4],atol=1e-12)
    group,block=geometry['panel'].split('_',1)
    for a in [x for x in m['artists'] if x['panel']==geometry['panel']]:
        if a['kind']=='points':
            z=points.loc[points.group.eq(group)&points.block.eq(block)&points.ABL.eq(a['ABL'])].sort_values('ILD')
            for key in ['ILD','displayed_ILD','mean','lower95','upper95']: np.testing.assert_allclose(a[key],z[key],atol=1e-12)
            assert a['color']==m['colors'][str(a['ABL'])]
            shown_points.extend((group,block,a['ABL'],i) for i in a['ILD'])
            low,high=min(a['lower95']),max(a['upper95'])
        else:
            assert a['ABL']=='all' if block=='gamma' else a['ABL'] in ABLS
            abl=20 if a['ABL']=='all' else a['ABL']
            z=curves.loc[curves.group.eq(group)&curves.ABL.eq(abl)].sort_values('ILD')
            np.testing.assert_allclose(a['ILD'],z.ILD,atol=1e-14)
            np.testing.assert_allclose(a['values'],z[block],atol=1e-12)
            shown_curves.append((group,block,a['ABL']))
            low,high=min(a['values']),max(a['values'])
        assert low>geometry['ylim'][0] and high<geometry['ylim'][1]
assert len(shown_points)==len(set(shown_points))==120
assert len(shown_curves)==len(set(shown_curves))==8
for name,digest in m['artifacts'].items():
    with (OUT/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    if name.endswith('.png'):
        with Image.open(OUT/name) as im: assert abs(im.width-2000)<=1 and abs(im.height-2040)<=1
    if name.endswith('.pdf'): assert len(PdfReader(OUT/name).pages)==1
report=dict(status='passed',panels=4,direct_posterior_points=120,credible_intervals=120,continuous_curve_rows=1926,
    independent_mapped_values=3852,plotted_curves=8,maximum_independent_mapping_error=max_error,
    formula_agrees_at_all_60_fitted_conditions=True,Gamma_ABL_invariance_verified=True,
    exact_trial_populations_and_frozen_values_match=True,source_hashes_unchanged=len(m['source_hashes']),
    units_and_display_offsets_verified=True,all_points_intervals_and_curves_unclipped=True,artifacts=m['artifacts'])
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
