# %% Independent posterior, raw-trial, quadrature and figure verification
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad,trapezoid
from scipy.stats import invgauss
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_comparison'
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm
EXPECTED={'wt':(68184,5),'hom':(67715,6)}
ILDS=[-16,-8,-4,-2,-1,1,2,4,8,16]
manifest=json.loads((OUT/'plot_manifest.json').read_text())
parameters=pd.read_csv(OUT/'parameter_comparisons.csv',float_precision='round_trip')
signed=pd.read_csv(OUT/'signed_condition_means.csv',float_precision='round_trip')
points=pd.read_csv(OUT/'chronometric_points.csv',float_precision='round_trip')
weights=pd.read_csv(OUT/'pooling_weights.csv',float_precision='round_trip')
animal_table=pd.read_csv(OUT/'animal_ABL_abs_ILD_means.csv',float_precision='round_trip')
merged_table=pd.read_csv(OUT/'animal_ABL_pooled_abs_ILD_means.csv',float_precision='round_trip')
members=pd.read_csv(OUT/'membership.csv')
mapping=pd.read_csv(OUT/'condition_mapping.csv',float_precision='round_trip')
assert [len(x) for x in [parameters,signed,points,animal_table,merged_table]]==[184,60,40,165,55]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def independent_model_mean(onsets,gamma,omega,delay,w,fixed):
    """Adaptive integration of conditional survival, using independent SciPy CDFs.

    Primary plotting uses fixed-node integration of t*density. This reference
    instead integrates (unconditional survival - terminal survival) / mass.
    """
    v,a,nd,p,beta=fixed
    sf0=(1-p)*invgauss.sf(onsets-nd,mu=1/(v*a),scale=a*a)+p*np.exp(-beta*onsets)
    es1=1-scipy_ddm(1.,gamma,omega,w,delay,cumulative=True)
    sf1=(1-p)*invgauss.sf(onsets+1-nd,mu=1/(v*a),scale=a*a)*es1+p*np.exp(-beta*(onsets+1))
    norm=sf0-sf1
    assert (norm>0).all()
    def survival(rt):
        es=1-scipy_ddm(rt,gamma,omega,w,delay,cumulative=True)
        sf=(1-p)*invgauss.sf(onsets+rt-nd,mu=1/(v*a),scale=a*a)*es+p*np.exp(-beta*(onsets+rt))
        return float(np.mean((sf-sf1)/norm))
    np.testing.assert_allclose([survival(0),survival(1)],[1,0],atol=1e-12)
    value,error=quad(survival,0,1,points=[delay] if 0<delay<1 else None,
                     epsabs=1e-9,epsrel=1e-9,limit=150)
    return value,error


# %% Rebuild selection directly from the cleaned CSV; no repeat refiltering
raw=pd.read_csv(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',float_precision='round_trip',
    usecols=['filename','genotype','animal','training_level','session_type','success','ABL','ILD','timed_rt','intended_fix_time'])
raw['ABL_mapped']=raw.ABL.replace({59:60})
population={};model_reference={};maximum_model_error=0.;maximum_saved_cdf_error=0.;quad_errors=[]
for group,(count,n_animals) in EXPECTED.items():
    source=ROOT/f'asd_analysis/shank3_cohort1_{group}_gamma_omega'
    mask=(raw.filename.eq('SHANK3_cohort1.csv')&raw.genotype.eq(group)&raw.training_level.eq(16)
          &raw.session_type.eq(1)&raw.success.isin([1,-1])&np.isfinite(raw.timed_rt)&raw.timed_rt.between(0,1)
          &raw.ABL_mapped.isin([20,40,60])&raw.ILD.isin(ILDS))
    data=raw.loc[mask].copy();data['abs_ILD']=data.ILD.abs()
    assert len(data)==count and data.animal.nunique()==n_animals
    population[group]=data
    source_trials=pd.read_csv(source/'fit_trials.csv.gz',float_precision='round_trip')
    np.testing.assert_array_equal(data.index+2,source_trials.prepared_csv_line)
    np.testing.assert_array_equal(data.timed_rt,source_trials.timed_rt)
    np.testing.assert_array_equal(data.intended_fix_time/1000,source_trials.onset_s)
    m=members.loc[members.group.eq(group)].set_index('animal')
    assert set(m.index)==set(data.animal)
    for animal,z in data.groupby('animal'):
        assert m.loc[animal,'trials']==len(z)

    posterior=dict(np.load(source/'condition_delay/posterior_samples.npz'))
    frozen=np.asarray(json.loads((source/'fixed_proactive.json').read_text())['values'])
    cond=pd.read_csv(source/'conditions.csv',float_precision='round_trip').set_index('condition')
    expected_names=[]
    for block in ['gamma','omega','t_E_aff','w','del_go']:
        draws=posterior[block]
        assert draws.shape==(10000,30) if block in ['gamma','omega','t_E_aff'] else draws.shape==(10000,)
        for i in range(draws.shape[1] if draws.ndim>1 else 1):
            name=f'{block}_{i:02d}' if draws.ndim>1 else block
            expected_names.append(name)
            saved=parameters.loc[parameters.group.eq(group)&parameters.parameter.eq(name)]
            assert len(saved)==1
            row=saved.iloc[0]
            array=draws[:,i] if draws.ndim>1 else draws
            expected=[np.mean(array),*np.percentile(array,[2.5,97.5])]
            np.testing.assert_allclose(row[['mean','lower95','upper95']].astype(float),expected,atol=1e-13,rtol=1e-13)
            scale=1000 if block in ['t_E_aff','del_go'] else 1
            assert row.display_scale==scale
            np.testing.assert_allclose(row[['display_mean','display_lower95','display_upper95']].astype(float),np.array(expected)*scale,atol=1e-11)
            if draws.ndim>1:
                assert row.condition==i and row.ABL==cond.loc[i,'ABL'] and row.ILD==cond.loc[i,'ILD']
            else:
                assert pd.isna(row.condition) and pd.isna(row.ABL) and pd.isna(row.ILD)
            limits=manifest['parameter_ylim'][block]
            assert limits[0]<row.display_lower95<=row.display_mean<=row.display_upper95<limits[1]
    assert set(parameters.loc[parameters.group.eq(group),'parameter'])==set(expected_names)

    saved_cdf=pd.read_csv(source/'condition_delay/valid_cdfs_1ms.csv.gz',float_precision='round_trip')
    for i,row in cond.iterrows():
        z=data.loc[data.ABL_mapped.eq(row.ABL)&data.ILD.eq(row.ILD)]
        result=signed.loc[signed.group.eq(group)&signed.condition.eq(i)].iloc[0]
        cm=mapping.loc[mapping.group.eq(group)&mapping.condition.eq(i)].iloc[0]
        assert cm.ABL==row.ABL and cm.ILD==row.ILD and result.n==cm.trials==len(z)
        np.testing.assert_allclose(result.data_mean_RT_s,np.mean(z.timed_rt),atol=1e-14)
        value,error=independent_model_mean(z.intended_fix_time.to_numpy()/1000,
            posterior['gamma'][:,i].mean(),posterior['omega'][:,i].mean(),posterior['t_E_aff'][:,i].mean(),
            posterior['w'].mean(),frozen)
        np.testing.assert_allclose(value,result.model_mean_RT_s,atol=1e-6,rtol=0)
        f=saved_cdf.loc[saved_cdf.condition.eq(i)].sort_values('rt_s')
        cdf_mean=trapezoid(1-f.model_cdf,f.rt_s)
        np.testing.assert_allclose(cdf_mean,result.model_mean_from_saved_CDF_s,atol=1e-13)
        np.testing.assert_allclose(cdf_mean,result.model_mean_RT_s,atol=1e-6,rtol=0)
        np.testing.assert_allclose(result.integrated_mass,1,atol=2e-8)
        maximum_model_error=max(maximum_model_error,abs(value-result.model_mean_RT_s))
        maximum_saved_cdf_error=max(maximum_saved_cdf_error,abs(cdf_mean-result.model_mean_RT_s))
        quad_errors.append(error);model_reference[group,i]=value
    print(group+': all 92 posterior summaries and 30 independent model means passed',flush=True)

# %% Animal summaries, pooled dots, SEM denominators and signed/ABL weights
for table,include_abl in [(animal_table,True),(merged_table,False)]:
    for row in table.itertuples():
        data=population[row.group]
        mask=data.animal.eq(row.animal)&data.abs_ILD.eq(row.abs_ILD)
        if include_abl:
            mask &= data.ABL_mapped.eq(row.ABL)
        z=data.loc[mask]
        assert len(z)==row.n
        np.testing.assert_allclose(row.mean_RT_s,np.mean(z.timed_rt),atol=1e-14)

for row in points.itertuples():
    data=population[row.group]
    selected=data.loc[data.abs_ILD.eq(row.abs_ILD)]
    if row.view=='ABL_separate':
        selected=selected.loc[selected.ABL_mapped.eq(row.ABL)]
    else:
        assert row.view=='ABL_pooled' and pd.isna(row.ABL)
    animal_means=np.array([z.timed_rt.to_numpy().mean() for _,z in selected.groupby('animal')])
    n=len(animal_means)
    sd=np.sqrt(np.sum((animal_means-animal_means.mean())**2)/(n-1))
    assert row.n==len(selected) and row.animals==n==EXPECTED[row.group][1]
    np.testing.assert_allclose([row.data_mean_RT_s,row.unweighted_animal_mean_RT_s,row.animal_sample_SD_s,row.animal_SEM_s],
        [selected.timed_rt.mean(),animal_means.mean(),sd,sd/np.sqrt(n)],atol=1e-14)
    w=weights.loc[weights.group.eq(row.group)&weights.view.eq(row.view)&weights.abs_ILD.eq(row.abs_ILD)]
    if row.view=='ABL_separate':
        w=w.loc[w.ABL.eq(row.ABL)]
    assert len(w)==(2 if row.view=='ABL_separate' else 6) and w.n.sum()==len(selected)
    np.testing.assert_allclose(w.weight,w.n/len(selected),atol=1e-15)
    np.testing.assert_allclose(w.weight.sum(),1,atol=1e-15)
    reference=0.
    for component in w.itertuples():
        count=((selected.ABL_mapped==component.component_ABL)&(selected.ILD==component.signed_ILD)).sum()
        assert count==component.n
        source=signed.loc[signed.group.eq(row.group)&signed.condition.eq(component.condition)].iloc[0]
        assert source.ABL==component.component_ABL and source.ILD==component.signed_ILD
        reference+=component.weight*model_reference[row.group,component.condition]
    np.testing.assert_allclose(row.model_mean_RT_s,np.dot(w.weight,w.model_mean_RT_s),atol=1e-14)
    np.testing.assert_allclose(row.model_mean_RT_s,reference,atol=1e-6,rtol=0)
    assert 0<=row.data_mean_RT_s-row.animal_SEM_s<=row.data_mean_RT_s+row.animal_SEM_s<=.30
    assert 0<=row.model_mean_RT_s<=.30
for (group,view),table in points.groupby(['group','view']):
    assert table.n.sum()==EXPECTED[group][0]

# %% Display/artifact checks, hashes and independently verified manifest
assert manifest['group_colors']=={'wt':'#1f77b4','hom':'#d62728'}
assert manifest['ABL_colors']=={'20':'#1f77b4','40':'#ff7f0e','60':'#2ca02c'}
assert manifest['model_linewidth']==1. and manifest['chronometric_ylim']==[0,.3]
assert manifest['chronometric_panel_box_aspect']==1.
assert len(manifest['figures'])==7 and len(manifest['artifacts'])==14
for name,expected in manifest['artifacts'].items():
    assert digest(OUT/name)==expected,name
    if name.endswith('.pdf'):
        info=subprocess.check_output(['pdfinfo',str(OUT/name)],text=True)
        assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))==1
    else:
        with Image.open(OUT/name) as im:
            assert im.width>=1000 and im.height>=800
baseline=json.loads((OUT/'preservation_baseline.json').read_text())
for path,expected in baseline.items():
    assert digest(ROOT/path)==expected,path
assert digest(ROOT/manifest['source'])==manifest['source_sha256']
report={'status':'passed','parameter_means_and_intervals':184,'signed_condition_model_means':60,
    'chronometric_data_model_pairs':40,'available_animal_SEM_bars':40,
    'animal_ABL_abs_ILD_means':165,'animal_ABL_pooled_abs_ILD_means':55,
    'groups':manifest['groups'],'max_independent_model_mean_error_s':maximum_model_error,
    'max_model_vs_saved_CDF_mean_error_s':maximum_saved_cdf_error,
    'max_adaptive_quadrature_error_estimate_s':max(quad_errors),
    'posterior_quantiles_from_saved_draws_verified':True,'raw_trial_identities_verified':True,
    'signed_and_ABL_trial_weighting_verified':True,'animal_SEM_denominators_verified':True,
    'all_14_artifacts_and_seven_single_page_PDFs_verified':True,'input_and_fit_hashes_unchanged':True,
    'figure_artifact_hashes':manifest['artifacts'],'source_sha256':digest(Path(__file__)),
    'independent_reference_source':str((ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py').relative_to(ROOT)),
    'independent_reference_sha256':digest(ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py')}
(OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
