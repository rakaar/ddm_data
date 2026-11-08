# %% Imports and editable display/integration settings
"""Full parent-design RTDs and valid-window psychometrics at posterior means.

--preview-initial exercises the plotting/audit pipeline in a separate folder;
it never publishes initial estimates or overwrites a fit.
"""
import argparse
import json
import os
from pathlib import Path
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0,{int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.integrate import trapezoid
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp
from shank3_wt_gamma_omega_utils import ROOT,ABLS,ILDS,sha256,evidence_bound

OUT = ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
VARIANTS = ['condition_delay']
LABELS = {'shared_delay':'Shared reactive delay','condition_delay':'Condition-specific reactive delays'}
COLORS = {20:'#1f77b4',40:'#ff7f0e',60:'#2ca02c'}
ABS_ILDS = [1,2,4,8,16]
EDGES = np.linspace(-1,1,101)  # 20 ms; zero is an exact boundary.
GRID = np.linspace(-1,1,2001)  # 1 ms full RTD evaluation.
VALID_GRID = np.linspace(0,1,1001)
QUADRATURE_ORDER = 128
CHUNK = 128
DPI = 180
DATA_LINEWIDTH, MODEL_LINEWIDTH, PSYCH_LINEWIDTH = 1.4, .8, 1.0
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preview-initial',action='store_true')
args=parser.parse_args()
DEST=OUT/'initial_preview' if args.preview_initial else OUT
DEST.mkdir(exist_ok=True)
conditions=pd.read_csv(OUT/'conditions.csv',float_precision='round_trip')
parent=pd.read_csv(OUT/'parent_trials.csv.gz',float_precision='round_trip')
valid=pd.read_csv(OUT/'fit_trials.csv.gz',float_precision='round_trip')
fixed=np.array(json.loads((OUT/'fixed_proactive.json').read_text())['values'])
v,a,nd,p_lapse,beta=fixed
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def average_fixed(times, onsets, weights=None):
    """Exact trial average; optional weights are 1/Z(c), not re-normalized.

No onset rounding, histogramming, subsampling or animal averaging is used.
Chunk only the evaluation-time axis to keep intermediate arrays modest.
"""
    times=np.asarray(times)
    weights=np.ones(len(onsets)) if weights is None else np.asarray(weights)
    result={k:np.empty(len(times)) for k in ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']}
    for start in range(0,len(times),CHUNK):
        sl=slice(start,start+CHUNK)
        t=times[sl,None]+np.asarray(onsets)[None,:]
        quantities=[invgauss.pdf(t-nd,mu=1/(v*a),scale=a*a),
                    invgauss.sf(t-nd,mu=1/(v*a),scale=a*a),
                    np.where(t>=0,beta*np.exp(-beta*np.maximum(t,0)),0.),
                    np.exp(-beta*np.maximum(t,0))]
        for key,quantity in zip(result,quantities):
            result[key][sl]=(quantity*weights[None,:]).mean(axis=1)
    return result


def evidence(times,gamma,omega,delay,w,go):
    """Evidence PDF/survival and right readout for each requested time."""
    pr,fr,sr=[np.asarray(x) for x in evidence_bound(jnp.asarray(times)-delay,gamma,omega,w,1)]
    pl,fl,sl=[np.asarray(x) for x in evidence_bound(jnp.asarray(times)-delay,gamma,omega,w,-1)]
    _,fgr,sgr=[np.asarray(x) for x in evidence_bound(jnp.asarray(times)-delay+go,gamma,omega,w,1)]
    _,_,sgl=[np.asarray(x) for x in evidence_bound(jnp.asarray(times)-delay+go,gamma,omega,w,-1)]
    return pr+pl,sr+sl,pr,.5*(sgr+sgl)+fgr-fr


# %% Common empirical bins and exact parent timing averages, reused by both fits
display=parent.loc[parent.fit_valid|parent.display_abort].copy()
display['display_rt']=np.where(display.fit_valid,display.timed_rt,display.abort_rt_s)
assert len(display)==80989 and display.fit_valid.sum()==67715 and display.display_abort.sum()==13274
data_rows=[]
for abl in ABLS:
    for ild in ABS_ILDS:
        z=display.loc[display.ABL_mapped.eq(abl)&display.ILD.abs().eq(ild)]
        population=parent.loc[parent.ABL_mapped.eq(abl)&parent.ILD.abs().eq(ild)]
        counts,_=np.histogram(z.display_rt,EDGES)
        assert counts.sum()==len(z)
        for i,count in enumerate(counts):
            data_rows.append(dict(ABL=abl,abs_ILD=ild,bin=i,left=EDGES[i],right=EDGES[i+1],count=count,
                density=count/len(z)/.020,display_trials=len(z),parent_trials=len(population),
                observed_window_fraction=len(z)/len(population)))
data_bins=pd.DataFrame(data_rows)
data_bins.to_csv(DEST/'rtd_data_bins.csv',index=False)
cache_path=OUT/'fixed_parent_design_averages.npz'
cache_sources={name:sha256(OUT/name) for name in ['parent_trials.csv.gz','fixed_proactive.json']}
if cache_path.exists():
    assert json.loads((OUT/'fixed_parent_design_manifest.json').read_text())['sources']==cache_sources
    cache=dict(np.load(cache_path))
    np.testing.assert_array_equal(cache.pop('grid'),GRID)
else:
    cache={}
    for row in conditions.itertuples():
        z=parent.loc[parent.condition.eq(row.condition)]
        for key,value in average_fixed(GRID,z.onset_s.to_numpy()).items():
            cache[f'{row.condition}_{key}']=value
    np.savez_compressed(cache_path,grid=GRID,**cache)
    (OUT/'fixed_parent_design_manifest.json').write_text(json.dumps({'sources':cache_sources,
        'grid_step_s':.001,'onset_average':'every exact parent-trial onset, equal trial weight'},indent=2)+'\n')

# %% Prediction arrays: parent-weighted RTDs and individually normalized choices
comparison=[]
manifests=[]
for variant in VARIANTS:
    folder=DEST/variant
    folder.mkdir(exist_ok=True)
    if args.preview_initial:
        gamma=conditions.gamma_init.to_numpy();omega=conditions.omega_init.to_numpy()
        delay=np.full(30,.05);w=.5;go=.08
        run={'status':'INITIAL PREVIEW — not fitted','parameter_count':63 if variant=='shared_delay' else 92}
    else:
        run=json.loads((OUT/variant/'run_summary.json').read_text())
        samples=dict(np.load(OUT/variant/'posterior_samples.npz'))
        gamma,omega=samples['gamma'].mean(0),samples['omega'].mean(0)
        delay=np.broadcast_to(samples['t_E_aff'].mean(0),(30,))
        w,go=float(samples['w'].mean()),float(samples['del_go'].mean())
    parent_rows=[];valid_rows=[];points=[];parameters=[];integration_errors=[]
    for row in conditions.itertuples():
        i=row.condition
        g,o,d=gamma[i],omega[i],delay[i]
        parameters.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,gamma=g,omega=o,t_E_aff=d,w=w,del_go=go))
        e_pdf,e_sf,_,_=evidence(GRID,g,o,d,w,go)
        ap,ass,lp,ls=[cache[f'{i}_{key}'] for key in ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']]
        pdf=(1-p_lapse)*(ap*e_sf+ass*e_pdf)+p_lapse*lp
        cdf=1-(1-p_lapse)*ass*e_sf-p_lapse*ls
        for t,q,f in zip(GRID,pdf,cdf):
            parent_rows.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,rt_s=t,parent_trials=row.parent_trials,
                unnormalized_density=q,unconditional_cdf=f))

        z=valid.loc[valid.condition.eq(i)]
        onset=z.onset_s.to_numpy()
        sa0=invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a)
        sa1=invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a)
        se1=evidence(np.array([1.]),g,o,d,w,go)[1][0]
        norm=(1-p_lapse)*(sa0-sa1*se1)+p_lapse*(np.exp(-beta*onset)-np.exp(-beta*(onset+1)))
        assert (norm>0).all() and (norm<=1).all()
        # Quadrature splits at both evidence/readout onsets; two orders verify
        # the integrated probability rather than relying on the plot grid.
        integrals=[]
        cuts=sorted(set([0.,1.]+[x for x in [d-go,d] if 0<x<1]))
        for order in [QUADRATURE_ORDER,2*QUADRATURE_ORDER]:
            nodes,weights=leggauss(order)
            times=np.concatenate([(hi-lo)/2*nodes+(hi+lo)/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
            quadrature=np.concatenate([(hi-lo)/2*weights for lo,hi in zip(cuts[:-1],cuts[1:])])
            avg=average_fixed(times,onset,1/norm)
            ep,es,pr,readout=evidence(times,g,o,d,w,go)
            qr=(1-p_lapse)*(avg['proactive_pdf']*readout+avg['proactive_survival']*pr)+.5*p_lapse*avg['lapse_pdf']
            qt=(1-p_lapse)*(avg['proactive_pdf']*es+avg['proactive_survival']*ep)+p_lapse*avg['lapse_pdf']
            integrals.append([float(quadrature@qr),float(quadrature@qt)])
        np.testing.assert_allclose(integrals[0],integrals[1],atol=2e-8,rtol=2e-8)
        np.testing.assert_allclose(integrals[-1][1],1.,atol=2e-8)
        integration_errors.append(float(np.max(np.abs(np.array(integrals[0])-integrals[1]))))
        avg=average_fixed(VALID_GRID,onset,1/norm)
        ep,es,_,_=evidence(VALID_GRID,g,o,d,w,go)
        model_cdf=(1-p_lapse)*((sa0/norm).mean()-avg['proactive_survival']*es)+p_lapse*((np.exp(-beta*onset)/norm).mean()-avg['lapse_survival'])
        empirical_cdf=np.searchsorted(np.sort(z.timed_rt.to_numpy()),VALID_GRID,side='right')/len(z)
        for t,f,emp in zip(VALID_GRID,model_cdf,empirical_cdf):
            valid_rows.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,rt_s=t,model_cdf=f,data_cdf=emp))
        points.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,trials=len(z),right=row.right,left=row.left,
            data_p_right=row.p_right,model_p_right=integrals[-1][0],integrated_both_choices=integrals[-1][1],
            min_trial_window_mass=norm.min(),mean_trial_window_mass=norm.mean(),max_trial_window_mass=norm.max(),
            rt_cdf_max_abs_gap_1ms=np.abs(model_cdf-empirical_cdf).max(),
            rt_cdf_integrated_abs_gap=trapezoid(np.abs(model_cdf-empirical_cdf),VALID_GRID)))
        print(f'{variant}: condition {i+1}/30 numerical diagnostics complete',flush=True)
    parent_predictions=pd.DataFrame(parent_rows)
    parent_predictions.to_csv(folder/'signed_parent_model_1ms.csv.gz',index=False,compression='gzip')
    pd.DataFrame(valid_rows).to_csv(folder/'valid_cdfs_1ms.csv.gz',index=False,compression='gzip')
    pd.DataFrame(parameters).to_csv(folder/'condition_parameter_means.csv',index=False)
    psych=pd.DataFrame(points)
    psych.to_csv(folder/'psychometric_predictions.csv',index=False)

    # Pool the signed conditions BEFORE normalizing, using their parent counts.
    curve_rows=[];bin_rows=[];panel_rows=[]
    for abl in ABLS:
        for ild in ABS_ILDS:
            z=parent_predictions.loc[parent_predictions.ABL.eq(abl)&parent_predictions.ILD.abs().eq(ild)]
            signed=[z.loc[z.ILD.eq(sign*ild)].sort_values('rt_s') for sign in [-1,1]]
            weights=np.array([x.parent_trials.iloc[0] for x in signed],dtype=float)
            q=np.average([x.unnormalized_density.to_numpy() for x in signed],axis=0,weights=weights)
            f=np.average([x.unconditional_cdf.to_numpy() for x in signed],axis=0,weights=weights)
            mass=f[-1]-f[0]
            normalized=q/mass
            empirical=data_bins.loc[data_bins.ABL.eq(abl)&data_bins.abs_ILD.eq(ild)]
            bin_masses=np.diff(f[::20])
            for t,rawq,density,cdf in zip(GRID,q,normalized,(f-f[0])/mass):
                curve_rows.append(dict(ABL=abl,abs_ILD=ild,rt_s=t,unnormalized_density=rawq,density=density,window_cdf=cdf))
            for b,m in enumerate(bin_masses):
                bin_rows.append(dict(ABL=abl,abs_ILD=ild,bin=b,left=EDGES[b],right=EDGES[b+1],
                    unnormalized_mass=m,window_mass=m/mass,density=m/mass/.02))
            panel_rows.append(dict(ABL=abl,abs_ILD=ild,parent_trials=int(weights.sum()),
                negative_ILD_parent_trials=int(weights[0]),positive_ILD_parent_trials=int(weights[1]),
                display_trials=int(empirical.display_trials.iloc[0]),data_window_fraction=float(empirical.observed_window_fraction.iloc[0]),
                model_window_mass=mass,model_density_trapezoid_area=trapezoid(normalized,GRID),
                model_exact_bin_area=bin_masses.sum()/mass,data_histogram_area=(empirical.density*.02).sum()))
    pd.DataFrame(curve_rows).to_csv(folder/'pooled_rtd_model_1ms.csv.gz',index=False,compression='gzip')
    pd.DataFrame(bin_rows).to_csv(folder/'pooled_rtd_model_bins.csv',index=False)
    panels=pd.DataFrame(panel_rows)
    panels.to_csv(folder/'rtd_panel_audit.csv',index=False)
    np.testing.assert_allclose(panels.model_density_trapezoid_area,1,atol=3e-4)
    np.testing.assert_allclose(panels.model_exact_bin_area,1,atol=1e-12)
    comparison.append(dict(variant=variant,status=run['status'],parameters=run['parameter_count'],valid_trials=len(valid),
        joint_log_likelihood=run.get('posterior_mean_joint_log_likelihood',np.nan),
        rt_log_likelihood=run.get('posterior_mean_rt_log_likelihood',np.nan),
        psychometric_rmse=np.sqrt(np.mean((psych.model_p_right-psych.data_p_right)**2)),
        trial_weighted_psychometric_rmse=np.sqrt(np.average((psych.model_p_right-psych.data_p_right)**2,weights=psych.trials)),
        mean_rt_cdf_max_gap_1ms=psych.rt_cdf_max_abs_gap_1ms.mean(),
        worst_rt_cdf_max_gap_1ms=psych.rt_cdf_max_abs_gap_1ms.max(),
        trial_weighted_rt_cdf_integrated_gap=np.average(psych.rt_cdf_integrated_abs_gap,weights=psych.trials)))
    manifests.append({'variant':variant,'status':run['status'],'rtd_panels':15,'psychometric_points':30,
        'quadrature_max_order_difference':max(integration_errors),'parent_onsets':'all exact parent-trial onsets',
        'psychometric_onsets':'all exact fitted valid-trial onsets; normalize each trial before averaging'})

# %% Common axes; two primary figures and one supplementary convergence figure
max_density=max(data_bins.density.max(),*(pd.read_csv(DEST/x/'pooled_rtd_model_1ms.csv.gz').density.max() for x in VARIANTS))
ylim=(0,float(np.ceil(max_density*1.08)))
artifacts=[]
for variant in VARIANTS:
    folder=DEST/variant
    curves=pd.read_csv(folder/'pooled_rtd_model_1ms.csv.gz')
    panels=pd.read_csv(folder/'rtd_panel_audit.csv')
    fig,axes=plt.subplots(3,5,figsize=(18,10),sharex=True,sharey=True)
    for ri,abl in enumerate(ABLS):
        for ci,ild in enumerate(ABS_ILDS):
            ax=axes[ri,ci]
            z=data_bins.loc[data_bins.ABL.eq(abl)&data_bins.abs_ILD.eq(ild)]
            model_curve=curves.loc[curves.ABL.eq(abl)&curves.abs_ILD.eq(ild)]
            ax.stairs(z.density,EDGES,color=COLORS[abl],linewidth=DATA_LINEWIDTH,zorder=3)
            ax.plot(model_curve.rt_s,model_curve.density,color='black',linewidth=MODEL_LINEWIDTH,zorder=2)
            ax.axvline(0,color='.55',linestyle=':',linewidth=.8)
            ax.set(xlim=(-1,1),ylim=ylim,xticks=[-1,-.5,0,.5,1])
            ax.set_title(f'|ILD| = {ild} dB   ·   n={int(z.display_trials.iloc[0]):,}',fontsize=10)
            if ci==0:
                ax.set_ylabel(f'ABL {abl} dB\nDensity (s⁻¹)')
            if ri==2:
                ax.set_xlabel('RT from stimulus onset (s)')
    title=LABELS[variant]
    prefix='INITIAL PREVIEW — ' if args.preview_initial else ''
    fig.suptitle(f'{prefix}SHANK3 cohort 1 HOM | {title}\n6 animals; 67,715 fitted choices + 13,274 displayed pre-onset aborts',fontsize=15,y=.985)
    fig.legend([Line2D([0],[0],color=COLORS[20],linewidth=DATA_LINEWIDTH),
        Line2D([0],[0],color='black',linewidth=MODEL_LINEWIDTH)],
        ['Data (ABL color; 20 ms bins)','Full race + fixed exponential lapse (1 ms evaluation)'],ncol=2,loc='upper center',bbox_to_anchor=(.5,.925),frameon=False)
    fig.text(.5,.025,'Each panel pools ±ILD trials, then normalizes data and model separately over [−1,1] s.\n'
        'Model averages all eligible parent-trial onset times and signed-condition frequencies. Negative aborts are shown only; they are not refitted.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.86,bottom=.12,hspace=.31,wspace=.16)
    stem=f'SHANK3_cohort1_HOM_{variant}_RTDs'
    for ext in ['png','pdf']:
        fig.savefig(DEST/f'{stem}.{ext}',dpi=DPI)
        artifacts.append(f'{stem}.{ext}')
    plt.close(fig)

    psych=pd.read_csv(folder/'psychometric_predictions.csv')
    fig,ax=plt.subplots(figsize=(8,6.7))
    for abl in ABLS:
        z=psych.loc[psych.ABL.eq(abl)].sort_values('ILD')
        ax.scatter(z.ILD,z.data_p_right,color=COLORS[abl],s=36,zorder=3,label=f'ABL {abl} dB (n={z.trials.sum():,})')
        ax.plot(z.ILD,z.model_p_right,color=COLORS[abl],linewidth=PSYCH_LINEWIDTH)
    ax.set(xlim=(-18,18),ylim=(0,1),xticks=[-16,-8,-4,0,4,8,16],xlabel='Signed ILD (dB)',ylabel='P(right)')
    ax.legend(frameon=False,loc='upper left')
    ax.set_title(f'{prefix}SHANK3 cohort 1 HOM | {title}\n67,715 valid 0–1 s choices; 6 animals',fontsize=13,pad=16)
    fig.text(.5,.035,'Dots: observed right choices / valid choices. Lines connect model predictions at the ten fitted ILDs.\n'
        'Model integrates right-choice density over [0,1] s, divides by each trial’s complete window mass,\n'
        'then averages over that condition’s exact valid-trial onset times. No sigmoid is fitted.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.12,right=.97,top=.86,bottom=.19)
    stem=f'SHANK3_cohort1_HOM_{variant}_psychometric'
    for ext in ['png','pdf']:
        fig.savefig(DEST/f'{stem}.{ext}',dpi=DPI)
        artifacts.append(f'{stem}.{ext}')
    plt.close(fig)

pd.DataFrame(comparison).to_csv(DEST/'in_sample_comparison.csv',index=False)
if not args.preview_initial:
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),squeeze=False)
    for i,variant in enumerate(VARIANTS):
        z=pd.read_csv(OUT/variant/'convergence.csv')
        run=json.loads((OUT/variant/'run_summary.json').read_text())
        x=z.step/1000
        axes[i,0].plot(x,z.mean_negative_elbo,color=COLORS[20 if i==0 else 40])
        axes[i,0].axvline(run['restored_best_step']/1000,color='black',linestyle='--',label='Restored best')
        axes[i,0].axvline(run['final_checked_step']/1000,color='.5',linestyle=':',label='Final checked')
        axes[i,0].set(title=f'{LABELS[variant]} | {run["status"]}',ylabel='Window mean negative ELBO',xlabel='Updates (thousands)')
        axes[i,0].legend(frameon=False)
        axes[i,1].semilogy(x,np.maximum(z.max_mean_change_sd,1e-8),color=COLORS[20 if i==0 else 40])
        axes[i,1].axhline(.1,color='black',linestyle='--')
        axes[i,1].set(xlabel='Updates (thousands)',ylabel='Largest parameter mean range / SD',title='Parameter stability (diagnostic only)')
    fig.suptitle('Full-rank SVI | one seed-0 start per model | 1,000-step checks')
    fig.tight_layout(rect=(0,0,1,.96))
    stem='SHANK3_cohort1_HOM_gamma_omega_convergence'
    for ext in ['png','pdf']:
        fig.savefig(DEST/f'{stem}.{ext}',dpi=DPI)
        artifacts.append(f'{stem}.{ext}')
    plt.close(fig)
manifest={'status':'initial_preview' if args.preview_initial else 'generated_pending_independent_verification',
    'fit_trials':67715,'display_trials':80989,'parent_trials':81721,'animals':6,'rtd_bin_width_s':.02,
    'data_linewidth':DATA_LINEWIDTH,'model_linewidth':MODEL_LINEWIDTH,'psychometric_linewidth':PSYCH_LINEWIDTH,
    'data_zorder':3,'model_zorder':2,'model_grid_width_s':.001,'display_xlim':[-1,1],'display_ylim':list(ylim),'ABL_colors':COLORS,
    'models':manifests,'artifacts':{name:sha256(DEST/name) for name in artifacts},
    'source_hashes':{str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))},
    'comparison':'in-sample, posterior-mean plug-in; no held-out evidence or initialization-sensitivity test',
    'uncertainty':'conditional on frozen proactive/lapse posterior means'}
(DEST/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved two primary figures, numerical predictions and in-sample comparisons:',DEST,flush=True)
