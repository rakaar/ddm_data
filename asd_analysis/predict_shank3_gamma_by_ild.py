# %% Prediction settings: no empirical arrays or original fits are overwritten
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import jax.numpy as jnp
from shank3_gamma_by_ild_utils import ROOT,OUT,ABLS,ILDS,PREVIOUS,VARIANTS,EXPECTED,sha256,expand_parameters,parameter_names
from shank3_tied_omega_prediction_utils import GRID,VALID_GRID,EDGES,ABS_ILDS,evidence_terms,valid_predictions

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initial-preview',action='store_true',help='Isolated initial values; never fitted results or publication inputs.')
args=parser.parse_args()
DEST=OUT/'initial_preview' if args.initial_preview else OUT
DEST.mkdir(exist_ok=True)
data_points=pd.read_csv(ROOT/'asd_analysis/shank3_cohort1_wt_hom_comparison/chronometric_points.csv',float_precision='round_trip')
parameter_rows=[];signed_rows=[];all_points=[];comparisons=[]
for group in EXPECTED:
    folder=OUT/group
    conditions=pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    valid=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    fixed=np.array(json.loads((folder/'fixed_proactive.json').read_text())['values'])
    p=fixed[3]
    cache=dict(np.load(folder/'fixed_parent_design_averages.npz'))
    np.testing.assert_array_equal(cache.pop('grid'),GRID)
    cache_manifest=json.loads((folder/'fixed_parent_design_manifest.json').read_text())
    for name,expected in cache_manifest['sources'].items():
        assert sha256(folder/name)==expected
    data_bins=pd.read_csv(folder/'rtd_data_bins.csv',float_precision='round_trip')
    for variant in VARIANTS:
        target=DEST/group/variant/'predictions'
        target.mkdir(parents=True,exist_ok=True)
        config=json.loads((folder/variant/'prepared_configuration.json').read_text())
        if args.initial_preview:
            free={k:jnp.asarray(v) for k,v in config['initial'].items()}
            samples=None;run={'status':'INITIAL VALUES ONLY — NOT FITTED'}
        else:
            run=json.loads((folder/variant/'run_summary.json').read_text())
            assert not run['smoke_test'] and run['posterior_draws']==10000
            samples=dict(np.load(folder/variant/'posterior_samples.npz'))
            free={k:jnp.asarray(v.mean(0)) for k,v in samples.items()}
        expanded={k:np.asarray(v) for k,v in expand_parameters(free,variant).items()}
        for block,value in free.items():
            flat=np.asarray(value).reshape(-1)
            for i,mean in enumerate(flat):
                draws=np.asarray(samples[block]).reshape(10000,-1)[:,i] if samples is not None else None
                lower,upper=np.quantile(draws,[.025,.975]) if draws is not None else (mean,mean)
                condition=None
                name=(f'gamma_{i:02d}' if block=='gamma' else f'omega_ABL_{ABLS[i]}' if block=='omega' and variant=='gamma_by_ild' else block)
                scale=1000 if block in ['t_E_aff','del_go'] else 1
                parameter_rows.append(dict(group=group,variant=variant,parameter=name,block=block,
                    ABL=condition.ABL if condition is not None else ABLS[i] if block=='omega' and variant=='gamma_by_ild' else np.nan,
                    ILD=ILDS[i] if block=='gamma' else np.nan,
                    mean=float(mean),lower95=float(lower),upper95=float(upper),
                    display_mean=float(mean*scale),display_lower95=float(lower*scale),display_upper95=float(upper*scale),
                    unit='ms' if scale==1000 else 's^-1' if block=='omega' else 'dimensionless',
                    posterior_interval_available=samples is not None))
        parent_rows=[];valid_rows=[];psych_rows=[];onset_rows=[];condition_rows=[]
        for row in conditions.itertuples():
            i=row.condition
            g,o,d=(float(expanded[k][i]) for k in ['gamma','omega','t_E_aff'])
            w,go=float(expanded['w']),float(expanded['del_go'])
            ep,es,_,_=map(np.asarray,evidence_terms(jnp.asarray(GRID),g,o,d,w,go))
            ap,ass,lp,ls=[cache[f'{i}_{key}'] for key in ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']]
            q=(1-p)*(ap*es+ass*ep)+p*lp
            f=1-(1-p)*ass*es-p*ls
            parent_rows.extend(dict(condition=i,ABL=row.ABL,ILD=row.ILD,rt_s=t,parent_trials=row.parent_trials,
                                    unnormalized_density=pdf,unconditional_cdf=cdf) for t,pdf,cdf in zip(GRID,q,f))
            z=valid.loc[valid.condition.eq(i)]
            result=valid_predictions(z.onset_s.to_numpy(),fixed,g,o,d,w,go)
            empirical=np.searchsorted(np.sort(z.timed_rt.to_numpy()),VALID_GRID,side='right')/len(z)
            valid_rows.extend(dict(condition=i,ABL=row.ABL,ILD=row.ILD,rt_s=t,model_cdf=cdf,data_cdf=emp,model_density=pdf)
                              for t,cdf,emp,pdf in zip(VALID_GRID,result['cdf'],empirical,result['density']))
            onset_rows.extend(dict(condition=i,ABL=row.ABL,ILD=row.ILD,onset_s=c,count=int(n),window_mass=mass)
                              for c,n,mass in zip(result['onsets'],result['onset_counts'],result['window_mass']))
            psych_rows.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,trials=len(z),right=row.right,left=row.left,
                data_p_right=row.p_right,model_p_right=result['p_right'],integrated_both_choices=result['integrated_mass'],
                rt_cdf_max_abs_gap_1ms=float(np.abs(result['cdf']-empirical).max()),
                rt_cdf_integrated_abs_gap=float(trapezoid(np.abs(result['cdf']-empirical),VALID_GRID))))
            signed_rows.append(dict(group=group,variant=variant,condition=i,ABL=row.ABL,ILD=row.ILD,abs_ILD=abs(row.ILD),
                n=len(z),data_mean_RT_s=z.timed_rt.mean(),model_mean_RT_s=result['mean_rt_s'],
                model_mean_from_saved_CDF_s=result['cdf_mean_s'],integrated_mass=result['integrated_mass'],
                quadrature_order_difference=result['quad_order_difference']))
            condition_rows.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,gamma=g,omega=o,t_E_aff=d,w=w,del_go=go))
        parent_predictions=pd.DataFrame(parent_rows)
        parent_predictions.to_csv(target/'signed_parent_model_1ms.csv.gz',index=False,compression='gzip',float_format='%.17g')
        pd.DataFrame(valid_rows).to_csv(target/'valid_cdfs_1ms.csv.gz',index=False,compression='gzip',float_format='%.17g')
        pd.DataFrame(onset_rows).to_csv(target/'valid_onset_weights.csv.gz',index=False,compression='gzip',float_format='%.17g')
        pd.DataFrame(condition_rows).to_csv(target/'condition_parameter_means.csv',index=False,float_format='%.17g')
        psych=pd.DataFrame(psych_rows)
        psych.to_csv(target/'psychometric_predictions.csv',index=False,float_format='%.17g')
        curve_rows=[];bin_rows=[];panel_rows=[]
        for abl in ABLS:
            for ild in ABS_ILDS:
                z=parent_predictions.loc[parent_predictions.ABL.eq(abl)&parent_predictions.ILD.abs().eq(ild)]
                sides=[z.loc[z.ILD.eq(sign*ild)].sort_values('rt_s') for sign in [-1,1]]
                weights=np.array([side.parent_trials.iloc[0] for side in sides])
                q=np.average([side.unnormalized_density for side in sides],weights=weights,axis=0)
                f=np.average([side.unconditional_cdf for side in sides],weights=weights,axis=0)
                mass=f[-1]-f[0]
                assert mass>0
                curve_rows.extend(dict(ABL=abl,abs_ILD=ild,rt_s=t,unnormalized_density=pdf,density=pdf/mass,window_cdf=cdf)
                                  for t,pdf,cdf in zip(GRID,q,(f-f[0])/mass))
                bins=np.diff(f[::20])/mass
                bin_rows.extend(dict(ABL=abl,abs_ILD=ild,bin=i,left=EDGES[i],right=EDGES[i+1],window_mass=b,density=b/.02)
                                for i,b in enumerate(bins))
                area=float(trapezoid(q/mass,GRID))
                np.testing.assert_allclose(area,1,atol=3e-4)
                np.testing.assert_allclose(bins.sum(),1,atol=1e-12)
                panel_rows.append(dict(ABL=abl,abs_ILD=ild,negative_parent_trials=int(weights[0]),positive_parent_trials=int(weights[1]),
                                       model_window_mass=mass,model_grid_area=area))
        pd.DataFrame(curve_rows).to_csv(target/'pooled_rtd_model_1ms.csv.gz',index=False,compression='gzip',float_format='%.17g')
        pd.DataFrame(bin_rows).to_csv(target/'pooled_rtd_model_bins.csv',index=False,float_format='%.17g')
        pd.DataFrame(panel_rows).to_csv(target/'rtd_panel_audit.csv',index=False,float_format='%.17g')
        comparisons.append(dict(group=group,variant=variant,status=run['status'],parameters=config['parameter_count'],
            joint_log_likelihood=run.get('posterior_mean_joint_log_likelihood',np.nan),
            psychometric_rmse=float(np.sqrt(np.mean((psych.data_p_right-psych.model_p_right)**2))),
            trial_weighted_psychometric_rmse=float(np.sqrt(np.average((psych.data_p_right-psych.model_p_right)**2,weights=psych.trials))),
            mean_rt_cdf_max_gap=float(psych.rt_cdf_max_abs_gap_1ms.mean()),
            trial_weighted_rt_cdf_integrated_gap=float(np.average(psych.rt_cdf_integrated_abs_gap,weights=psych.trials))))
        print(f'{group}/{variant}: 30 signed predictions and 15 pooled RTDs checked',flush=True)

# %% Reuse empirical pooled means/SEMs exactly; replace only fitted model means
signed=pd.DataFrame(signed_rows)
for variant in VARIANTS:
    for row in data_points.itertuples(index=False):
        z=signed.loc[signed.variant.eq(variant)&signed.group.eq(row.group)&signed.abs_ILD.eq(row.abs_ILD)]
        if row.view=='ABL_separate': z=z.loc[z.ABL.eq(row.ABL)]
        assert z.n.sum()==row.n
        record=row._asdict()
        record.update(variant=variant,model_mean_RT_s=float(np.average(z.model_mean_RT_s,weights=z.n)),
                      model_mean_from_saved_CDF_s=float(np.average(z.model_mean_from_saved_CDF_s,weights=z.n)))
        all_points.append(record)
pd.DataFrame(parameter_rows).to_csv(DEST/'parameter_comparisons.csv',index=False,float_format='%.17g')
signed.to_csv(DEST/'signed_condition_means.csv',index=False,float_format='%.17g')
pd.DataFrame(all_points).to_csv(DEST/'chronometric_points.csv',index=False,float_format='%.17g')
comparison=pd.DataFrame(comparisons)
if not args.initial_preview:
    reference=pd.read_csv(PREVIOUS/'in_sample_comparison.csv',float_precision='round_trip')
    reference=reference.loc[reference.variant.eq('omega_by_abl')].copy()
    reference['variant']='condition_gamma_omega_by_abl'
    comparison=pd.concat([comparison,reference],ignore_index=True)
comparison.to_csv(DEST/'in_sample_comparison.csv',index=False,float_format='%.17g')
assert len(parameter_rows)==32 and len(signed)==60 and len(all_points)==40
report=dict(status='initial_preview_only' if args.initial_preview else 'predictions_generated',
    initial_preview=args.initial_preview,source_sha256=sha256(Path(__file__)),
    parameter_estimates=32,signed_model_means=60,chronometric_pairs=40,empirical_values_unchanged=True,
    max_density_CDF_mean_difference_s=float(np.abs(signed.model_mean_RT_s-signed.model_mean_from_saved_CDF_s).max()))
(DEST/'prediction_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
