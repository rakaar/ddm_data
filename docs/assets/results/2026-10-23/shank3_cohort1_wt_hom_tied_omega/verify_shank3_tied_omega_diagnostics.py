# %% Independent SciPy predictions, pooling, plotted-artist and figure checks
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.integrate import quad,trapezoid
from shank3_tied_omega_utils import ROOT,OUT,ABLS,VARIANTS,EXPECTED,sha256
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initial-preview',action='store_true')
args=parser.parse_args()
DEST=OUT/'initial_preview' if args.initial_preview else OUT
signed=pd.read_csv(DEST/'signed_condition_means.csv',float_precision='round_trip')
points=pd.read_csv(DEST/'chronometric_points.csv',float_precision='round_trip')
parameters=pd.read_csv(DEST/'parameter_comparisons.csv',float_precision='round_trip')
empirical=pd.read_csv(ROOT/'asd_analysis/shank3_cohort1_wt_hom_comparison/chronometric_points.csv',float_precision='round_trip')
manifest=json.loads((DEST/'figures/plot_manifest.json').read_text())
with gzip.open(DEST/'figures/plotted_series.json.gz','rt') as stream:traces=json.load(stream)
assert manifest['initial_preview']==args.initial_preview and len(parameters)==140 and len(signed)==120 and len(points)==80
if not args.initial_preview:
    for (group,variant),table in parameters.groupby(['group','variant']):
        draws=dict(np.load(OUT/group/variant/'posterior_samples.npz'))
        for row in table.itertuples():
            index=int(row.parameter.rsplit('_',1)[1]) if row.block=='gamma' else ABLS.index(int(row.ABL)) if row.block=='omega' and variant=='omega_by_abl' else 0
            values=draws[row.block].reshape(10000,-1)[:,index]
            scale=1000 if row.block in ['t_E_aff','del_go'] else 1
            np.testing.assert_allclose([row.display_mean,row.display_lower95,row.display_upper95],
                                      np.r_[values.mean(),np.quantile(values,[.025,.975])]*scale,atol=1e-12)
mean_errors=[];choice_errors=[]
for group in EXPECTED:
    valid=pd.read_csv(OUT/group/'fit_trials.csv.gz',float_precision='round_trip')
    fixed=np.array(json.loads((OUT/group/'fixed_proactive.json').read_text())['values'])
    v,a,nd,p,beta=fixed
    cache=dict(np.load(OUT/group/'fixed_parent_design_averages.npz'))
    for variant in VARIANTS:
        folder=DEST/group/variant/'predictions'
        cond=pd.read_csv(folder/'condition_parameter_means.csv',float_precision='round_trip')
        parent=pd.read_csv(folder/'signed_parent_model_1ms.csv.gz',float_precision='round_trip')
        cdfs=pd.read_csv(folder/'valid_cdfs_1ms.csv.gz',float_precision='round_trip')
        psych=pd.read_csv(folder/'psychometric_predictions.csv',float_precision='round_trip')
        for row in cond.itertuples():
            data=valid.loc[valid.condition.eq(row.condition)]
            onset,count=np.unique(data.onset_s,return_counts=True);weight=count/len(data)
            g,o,d,w,go=row.gamma,row.omega,row.t_E_aff,row.w,row.del_go
            start=(1-p)*invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a)+p*np.exp(-beta*onset)
            end=(1-p)*invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a)*(1-scipy_ddm(1,g,o,w,d,cumulative=True))+p*np.exp(-beta*(onset+1))
            mass=start-end
            assert np.all(mass>0)
            normalized_weights=weight/mass
            def survival(rt):
                st=(1-p)*invgauss.sf(onset+rt-nd,mu=1/(v*a),scale=a*a)*(1-scipy_ddm(rt,g,o,w,d,cumulative=True))+p*np.exp(-beta*(onset+rt))
                return float((st-end)@normalized_weights)
            def right_density(rt):
                ap=invgauss.pdf(onset+rt-nd,mu=1/(v*a),scale=a*a)@normalized_weights
                ass=invgauss.sf(onset+rt-nd,mu=1/(v*a),scale=a*a)@normalized_weights
                lapse=(beta*np.exp(-beta*(onset+rt)))@normalized_weights
                pr=scipy_ddm(rt,g,o,w,d,side=1)
                fr=scipy_ddm(rt,g,o,w,d,cumulative=True,side=1)
                fgr=scipy_ddm(rt+go,g,o,w,d,cumulative=True,side=1)
                fgl=scipy_ddm(rt+go,g,o,w,d,cumulative=True,side=-1)
                readout=.5*(1-fgr-fgl)+fgr-fr
                return float((1-p)*(ap*readout+ass*pr)+.5*p*lapse)
            breaks=sorted(set(t for t in [d,d-go] if 0<t<1))
            mean=quad(survival,0,1,points=breaks,epsabs=1e-9,epsrel=1e-9)[0]
            pr=quad(right_density,0,1,points=breaks,epsabs=1e-9,epsrel=1e-9)[0]
            s=signed.loc[signed.group.eq(group)&signed.variant.eq(variant)&signed.condition.eq(row.condition)].iloc[0]
            prediction=psych.loc[psych.condition.eq(row.condition)].iloc[0]
            np.testing.assert_allclose(mean,s.model_mean_RT_s,atol=1e-6,rtol=0)
            np.testing.assert_allclose(pr,prediction.model_p_right,atol=2e-8,rtol=2e-8)
            np.testing.assert_allclose(prediction.data_p_right,data.response_poke.eq(1).mean(),atol=1e-14)
            curve=cdfs.loc[cdfs.condition.eq(row.condition)].sort_values('rt_s')
            np.testing.assert_allclose(trapezoid(1-curve.model_cdf,curve.rt_s),s.model_mean_RT_s,atol=1e-6,rtol=0)
            for t in [0.,.01,.05,.1,.2,.5,1.]:
                actual=curve.iloc[int(round(t*1000))].model_cdf
                np.testing.assert_allclose(actual,1-survival(t),atol=1e-10)
            pc=parent.loc[parent.condition.eq(row.condition)].sort_values('rt_s')
            age=pc.rt_s.to_numpy()
            ep=scipy_ddm(age,g,o,w,d)
            es=1-scipy_ddm(age,g,o,w,d,cumulative=True)
            ap,ass,lp,ls=[cache[f'{row.condition}_{key}'] for key in ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']]
            np.testing.assert_allclose(pc.unnormalized_density,(1-p)*(ap*es+ass*ep)+p*lp,atol=3e-10,rtol=2e-9)
            np.testing.assert_allclose(pc.unconditional_cdf,1-(1-p)*ass*es-p*ls,atol=3e-10)
            mean_errors.append(abs(mean-s.model_mean_RT_s));choice_errors.append(abs(pr-prediction.model_p_right))
        pooled=pd.read_csv(folder/'pooled_rtd_model_1ms.csv.gz',float_precision='round_trip')
        bins=pd.read_csv(folder/'pooled_rtd_model_bins.csv',float_precision='round_trip')
        for (abl,ild),curve in pooled.groupby(['ABL','abs_ILD']):
            sides=[parent.loc[parent.ABL.eq(abl)&parent.ILD.eq(sign*ild)].sort_values('rt_s') for sign in [-1,1]]
            weights=np.array([side.parent_trials.iloc[0] for side in sides])
            raw_q=np.average([side.unnormalized_density for side in sides],axis=0,weights=weights)
            raw_f=np.average([side.unconditional_cdf for side in sides],axis=0,weights=weights)
            mass=raw_f[-1]-raw_f[0]
            np.testing.assert_allclose(curve.density,raw_q/mass,atol=1e-12)
            b=bins.loc[bins.ABL.eq(abl)&bins.abs_ILD.eq(ild)]
            np.testing.assert_allclose(b.window_mass,np.diff(raw_f[::20])/mass,atol=1e-12)
            np.testing.assert_allclose(b.window_mass.sum(),1,atol=1e-12)
        print(group+'/'+variant+': independent SciPy moments/choices and all parent/pooled curves agree',flush=True)

# %% Actual artists must agree with their source tables, not just declared styles
for trace in traces:
    group,variant=trace['group'],trace['variant']
    role=trace['role'];abl=trace.get('ABL')
    folder=DEST/group/variant/'predictions'
    if role=='posterior':
        z=parameters.loc[parameters.group.eq(group)&parameters.variant.eq(variant)&parameters.block.eq(trace['block'])]
        if trace['block']=='gamma':z=z.loc[z.ABL.eq(abl)].sort_values('ILD')
        else:z=z.sort_values('ABL')
        np.testing.assert_allclose([trace['y'],trace['lower'],trace['upper']],
                                  z[['display_mean','display_lower95','display_upper95']].to_numpy().T,atol=1e-12)
        assert trace['color']=={'wt':'#1f77b4','hom':'#d62728'}[group]
    elif role.startswith('chronometric'):
        z=points.loc[points.variant.eq(variant)&points.group.eq(group)&points.view.eq('ABL_pooled' if abl is None else 'ABL_separate')]
        if abl is not None:z=z.loc[z.ABL.eq(abl)]
        z=z.sort_values('abs_ILD')
        np.testing.assert_array_equal(trace['y'],z.model_mean_RT_s if role.endswith('model') else z.data_mean_RT_s)
        if role.endswith('data'):
            np.testing.assert_allclose(trace['lower'],z.data_mean_RT_s-z.animal_SEM_s,atol=1e-14)
            np.testing.assert_allclose(trace['upper'],z.data_mean_RT_s+z.animal_SEM_s,atol=1e-14)
    elif role.startswith('psychometric'):
        z=pd.read_csv(folder/'psychometric_predictions.csv',float_precision='round_trip')
        z=z.loc[z.ABL.eq(abl)].sort_values('ILD')
        np.testing.assert_array_equal(trace['x'],z.ILD)
        np.testing.assert_array_equal(trace['y'],z.model_p_right if role.endswith('model') else z.data_p_right)
    else:
        model=role=='rtd_model'
        z=pd.read_csv(folder/'pooled_rtd_model_1ms.csv.gz' if model else OUT/group/'rtd_data_bins.csv',float_precision='round_trip')
        z=z.loc[z.ABL.eq(abl)&z.abs_ILD.eq(trace['abs_ILD'])].sort_values('rt_s' if model else 'bin')
        np.testing.assert_array_equal(trace['y'],z.density)
        np.testing.assert_array_equal(trace['x'] if model else trace['edges'],z.rt_s if model else np.r_[z.left,z.right.iloc[-1]])
        assert trace['linewidth']==(.7 if model else 1.4)
        if model:assert trace['linestyle']=='--'
for row in points.itertuples():
    old=empirical.loc[empirical.group.eq(row.group)&empirical.view.eq(row.view)&empirical.abs_ILD.eq(row.abs_ILD)]
    if row.view=='ABL_separate':old=old.loc[old.ABL.eq(row.ABL)]
    assert len(old)==1
    for name in ['n','animals','data_mean_RT_s','animal_SEM_s','animal_sample_SD_s','unweighted_animal_mean_RT_s']:
        np.testing.assert_array_equal(getattr(row,name),old.iloc[0][name])
    z=signed.loc[signed.group.eq(row.group)&signed.variant.eq(row.variant)&signed.abs_ILD.eq(row.abs_ILD)]
    if row.view=='ABL_separate':z=z.loc[z.ABL.eq(row.ABL)]
    np.testing.assert_allclose(row.model_mean_RT_s,np.average(z.model_mean_RT_s,weights=z.n),atol=1e-14)
assert len(manifest['figures'])==4
for figure in manifest['figures']:
    assert len(figure['panels'])==(7 if figure['kind']=='parameters' else 15)
    for panel in figure['panels']:
        pid=panel['panel']
        if '_rtd_' in pid:
            assert panel['xlim']==[-.1,.6] and panel['ylim']==manifest['RTD_ylim']
        elif figure['kind']=='behavior':
            np.testing.assert_allclose(panel['width_inches'],panel['height_inches'],atol=1e-12)
        relevant=[t for t in traces if t['figure']==figure['stem'] and t['panel']==pid]
        for trace in relevant:
            y=np.asarray(trace['y'])
            if 'lower' in trace:y=np.r_[y,trace['lower'],trace['upper']]
            assert y.min()>=panel['ylim'][0]-1e-12 and y.max()<=panel['ylim'][1]+1e-12
for name,expected in manifest['artifacts'].items():
    assert sha256(DEST/'figures'/name)==expected
    if name.endswith('.pdf'):
        info=subprocess.check_output(['pdfinfo',str(DEST/'figures'/name)],text=True)
        assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))==1
for name,expected in json.loads((OUT/'preservation_baseline.json').read_text()).items():
    assert sha256(ROOT/name)==expected,name
report=dict(status='passed',initial_preview=args.initial_preview,signed_model_means=120,psychometric_probabilities=120,
    posterior_parameters=140,chronometric_pairs=80,figures=4,parameter_panels_per_figure=7,behavior_panels_per_figure=15,
    max_independent_mean_error_s=max(mean_errors),max_independent_choice_probability_error=max(choice_errors),
    all_empirical_values_and_original_files_unchanged=True,figure_artifact_hashes=manifest['artifacts'],
    verifier_source_sha256=sha256(Path(__file__)))
(DEST/'diagnostic_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
