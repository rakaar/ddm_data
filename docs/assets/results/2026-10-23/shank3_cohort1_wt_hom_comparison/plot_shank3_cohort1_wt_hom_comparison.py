# %% Imports and editable source, integration and display settings
"""Seven comparisons from the saved condition-delay posteriors; no fitting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/shank3-wt-hom-comparison-mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.integrate import trapezoid
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_wt_gamma_omega_utils import ROOT,sha256,evidence_bound

OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_comparison'
FIT_ROOTS={g:ROOT/f'asd_analysis/shank3_cohort1_{g}_gamma_omega' for g in ['wt','hom']}
GROUP_COLORS={'wt':'#1f77b4','hom':'#d62728'}
ABL_COLORS={20:'#1f77b4',40:'#ff7f0e',60:'#2ca02c'}
LABELS={'wt':'WT','hom':'HOM'}
EXPECTED={'wt':(68184,5),'hom':(67715,6)}
ABLS=[20,40,60]
ILDS=[-16,-8,-4,-2,-1,1,2,4,8,16]
ABS_ILDS=[1,2,4,8,16]
BLOCKS={'gamma':('Gamma (dimensionless)',1.,'dimensionless'),
        'omega':('Omega (s⁻¹)',1.,'s^-1'),
        't_E_aff':('Reactive delay (ms)',1000.,'ms')}
DPI=180
MODEL_LINEWIDTH=1.0
CHRONOMETRIC_YLIM=(0,.30)
QUADRATURE_ORDERS=[128,256]
CHUNK=128
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def model_moments(onsets,gamma,omega,delay,w,go,fixed):
    """Complete marginal race+lapse density / exact per-trial valid-window mass.

    Average only after dividing by each onset's own window mass. The sum of
    the two choice densities cancels the del_go readout term exactly.
    """
    v,a,nd,p,beta=fixed
    sa0=invgauss.sf(onsets-nd,mu=1/(v*a),scale=a*a)
    sa1=invgauss.sf(onsets+1-nd,mu=1/(v*a),scale=a*a)
    se1=sum(float(evidence_bound(1-delay,gamma,omega,w,side)[2]) for side in [-1,1])
    norm=(1-p)*(sa0-sa1*se1)+p*(np.exp(-beta*onsets)-np.exp(-beta*(onsets+1)))
    assert (norm>0).all() and (norm<=1).all()
    cuts=sorted(set([0.,1.]+[t for t in [delay-go,delay] if 0<t<1]))
    results=[]
    for order in QUADRATURE_ORDERS:
        x,weights=leggauss(order)
        times=np.concatenate([(hi-lo)/2*x+(hi+lo)/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
        weights=np.concatenate([(hi-lo)/2*weights for lo,hi in zip(cuts[:-1],cuts[1:])])
        pdfs=[evidence_bound(times-delay,gamma,omega,w,side) for side in [-1,1]]
        ep=np.asarray(pdfs[0][0]+pdfs[1][0]);es=np.asarray(pdfs[0][2]+pdfs[1][2])
        density=np.empty(len(times))
        for start in range(0,len(times),CHUNK):
            sl=slice(start,start+CHUNK)
            absolute=times[sl,None]+onsets[None,:]
            fa=invgauss.pdf(absolute-nd,mu=1/(v*a),scale=a*a)
            sa=invgauss.sf(absolute-nd,mu=1/(v*a),scale=a*a)
            lapse=beta*np.exp(-beta*absolute)
            density[sl]=(((1-p)*(fa*es[sl,None]+sa*ep[sl,None])+p*lapse)/norm).mean(1)
        results.append((float(weights@density),float(weights@(times*density))))
    np.testing.assert_allclose(results[0],results[1],atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(results[-1][0],1,atol=2e-8)
    return dict(model_mean_RT_s=results[-1][1],integrated_mass=results[-1][0],
        quadrature_order_difference_s=abs(results[0][1]-results[1][1]),
        min_trial_window_mass=float(norm.min()),max_trial_window_mass=float(norm.max()))


def save_figure(fig,stem,description):
    for ext in ['png','pdf']:
        name=f'{stem}.{ext}'
        fig.savefig(OUT/name,dpi=DPI)
        artifacts[name]=sha256(OUT/name)
    plt.close(fig)
    figures.append({'stem':stem,'description':description})


# %% Preserve source files and reconstruct posterior summaries and signed means
baseline_path=OUT/'preservation_baseline.json'
if not baseline_path.exists():
    paths=[ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',
           ROOT/'asd_analysis/shank3_wt_gamma_omega_utils.py',ROOT/'asd_analysis/proactive_lapse_no_led_utils.py']
    for folder in FIT_ROOTS.values():
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    baseline={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    baseline_path.write_text(json.dumps(baseline,indent=2)+'\n')
baseline=json.loads(baseline_path.read_text())
for name,digest in baseline.items():
    assert sha256(ROOT/name)==digest,name

parameter_rows=[];signed_rows=[];members=[];animal_rows=[];merged_animal_rows=[]
conditions_all=[];trials={};groups={}
for group,folder in FIT_ROOTS.items():
    print('Source:',folder,flush=True)
    assert json.loads((folder/'independent_validation.json').read_text())['status']=='passed'
    samples=dict(np.load(folder/'condition_delay/posterior_samples.npz'))
    cond=pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    source_summary=pd.read_csv(folder/'condition_delay/posterior_summary.csv',float_precision='round_trip').set_index('parameter')
    fixed=np.asarray(json.loads((folder/'fixed_proactive.json').read_text())['values'])
    data=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    data['abs_ILD']=data.ILD.abs()
    assert len(data)==EXPECTED[group][0] and data.animal.nunique()==EXPECTED[group][1]
    assert data.success.isin([1,-1]).all() and np.isfinite(data.timed_rt).all() and data.timed_rt.between(0,1).all()
    trials[group]=data
    groups[group]={'trials':len(data),'animals':data.animal.nunique()}
    conditions_all.extend(cond.assign(group=group).to_dict('records'))
    for animal,z in data.groupby('animal'):
        members.append(dict(group=group,animal=animal,trials=len(z)))
    for keys,z in data.groupby(['animal','ABL_mapped','abs_ILD']):
        animal_rows.append(dict(group=group,animal=keys[0],ABL=keys[1],abs_ILD=keys[2],
            n=len(z),mean_RT_s=z.timed_rt.mean()))
    for keys,z in data.groupby(['animal','abs_ILD']):
        merged_animal_rows.append(dict(group=group,animal=keys[0],abs_ILD=keys[1],n=len(z),mean_RT_s=z.timed_rt.mean()))
    for block,array in samples.items():
        assert len(array)==10000 and np.isfinite(array).all()
        vectors=array[:,None] if array.ndim==1 else array
        for index in range(vectors.shape[1]):
            name=block if array.ndim==1 else f'{block}_{index:02d}'
            mean=float(vectors[:,index].mean());lo,hi=np.quantile(vectors[:,index],[.025,.975])
            np.testing.assert_allclose([mean,lo,hi],source_summary.loc[name,['mean','q025','q975']],atol=1e-12)
            scale=1000. if block in ['t_E_aff','del_go'] else 1.
            unit='ms' if scale==1000 else ('s^-1' if block=='omega' else 'dimensionless')
            condition=cond.loc[cond.condition.eq(index)].iloc[0] if array.ndim>1 else None
            parameter_rows.append(dict(group=group,parameter=name,block=block,
                condition=index if condition is not None else np.nan,
                ABL=condition.ABL if condition is not None else np.nan,
                ILD=condition.ILD if condition is not None else np.nan,
                mean=mean,lower95=lo,upper95=hi,display_scale=scale,display_unit=unit,
                display_mean=mean*scale,display_lower95=lo*scale,display_upper95=hi*scale))
    cdfs=pd.read_csv(folder/'condition_delay/valid_cdfs_1ms.csv.gz',float_precision='round_trip')
    for row in cond.itertuples():
        z=data.loc[data.condition.eq(row.condition)]
        i=row.condition
        values=model_moments(z.onset_s.to_numpy(),samples['gamma'][:,i].mean(),samples['omega'][:,i].mean(),
            samples['t_E_aff'][:,i].mean(),samples['w'].mean(),samples['del_go'].mean(),fixed)
        curve=cdfs.loc[cdfs.condition.eq(i)].sort_values('rt_s')
        cdf_mean=float(trapezoid(1-curve.model_cdf,curve.rt_s))
        np.testing.assert_allclose(values['model_mean_RT_s'],cdf_mean,atol=1e-6,rtol=0)
        signed_rows.append(dict(group=group,condition=i,ABL=row.ABL,ILD=row.ILD,abs_ILD=abs(row.ILD),
            n=len(z),right=int(z.response_poke.eq(1).sum()),data_mean_RT_s=z.timed_rt.mean(),
            model_mean_from_saved_CDF_s=cdf_mean,**values))
        print(f'{group}: signed condition {i+1}/30 mean checked',flush=True)

parameters=pd.DataFrame(parameter_rows);signed=pd.DataFrame(signed_rows)
animal_means=pd.DataFrame(animal_rows);merged_animal_means=pd.DataFrame(merged_animal_rows)
assert len(parameters)==184 and len(signed)==60 and len(animal_means)==165 and len(merged_animal_means)==55
parameters.to_csv(OUT/'parameter_comparisons.csv',index=False,float_format='%.17g')
signed.to_csv(OUT/'signed_condition_means.csv',index=False,float_format='%.17g')
animal_means.to_csv(OUT/'animal_ABL_abs_ILD_means.csv',index=False,float_format='%.17g')
merged_animal_means.to_csv(OUT/'animal_ABL_pooled_abs_ILD_means.csv',index=False,float_format='%.17g')
pd.DataFrame(members).to_csv(OUT/'membership.csv',index=False)
pd.DataFrame(conditions_all).to_csv(OUT/'condition_mapping.csv',index=False,float_format='%.17g')

# %% Actual-trial pooling; equal-animal SEM centered on pooled-trial mean
point_rows=[];weight_rows=[]
for group,data in trials.items():
    for view,levels in [('ABL_separate',ABLS),('ABL_pooled',[None])]:
        source=animal_means if view=='ABL_separate' else merged_animal_means
        for abl in levels:
            for ild in ABS_ILDS:
                mask=data.abs_ILD.eq(ild)
                cmask=signed.group.eq(group)&signed.abs_ILD.eq(ild)
                amask=source.group.eq(group)&source.abs_ILD.eq(ild)
                if abl is not None:
                    mask &= data.ABL_mapped.eq(abl)
                    cmask &= signed.ABL.eq(abl)
                    amask &= source.ABL.eq(abl)
                z=data.loc[mask];components=signed.loc[cmask];animals=source.loc[amask]
                assert components.n.sum()==len(z) and len(animals)==EXPECTED[group][1]
                weights=components.n/len(z)
                np.testing.assert_allclose(z.timed_rt.mean(),weights@components.data_mean_RT_s,atol=1e-14)
                sd=animals.mean_RT_s.std(ddof=1)
                point_rows.append(dict(group=group,view=view,ABL=abl,abs_ILD=ild,n=len(z),animals=len(animals),
                    data_mean_RT_s=z.timed_rt.mean(),unweighted_animal_mean_RT_s=animals.mean_RT_s.mean(),
                    animal_sample_SD_s=sd,animal_SEM_s=sd/np.sqrt(len(animals)),
                    model_mean_RT_s=weights@components.model_mean_RT_s,
                    model_mean_from_saved_CDF_s=weights@components.model_mean_from_saved_CDF_s))
                for component,weight in zip(components.itertuples(),weights):
                    weight_rows.append(dict(group=group,view=view,ABL=abl,abs_ILD=ild,
                        component_ABL=component.ABL,signed_ILD=component.ILD,condition=component.condition,
                        n=component.n,weight=weight,model_mean_RT_s=component.model_mean_RT_s))
points=pd.DataFrame(point_rows)
assert len(points)==40 and points.animal_SEM_s.notna().all()
assert ((points.data_mean_RT_s-points.animal_SEM_s)>=CHRONOMETRIC_YLIM[0]).all()
assert ((points.data_mean_RT_s+points.animal_SEM_s)<=CHRONOMETRIC_YLIM[1]).all()
assert points.model_mean_RT_s.between(*CHRONOMETRIC_YLIM).all()
points.to_csv(OUT/'chronometric_points.csv',index=False,float_format='%.17g')
pd.DataFrame(weight_rows).to_csv(OUT/'pooling_weights.csv',index=False,float_format='%.17g')

# %% Parameter comparisons: signed categories, common scale within each image
figures=[];artifacts={};parameter_limits={}
handles=[Line2D([],[],color=GROUP_COLORS[g],marker='o',linestyle='',label=LABELS[g]) for g in FIT_ROOTS]
for block,(ylabel,scale,unit) in BLOCKS.items():
    table=parameters.loc[parameters.block.eq(block)]
    low,high=table.display_lower95.min(),table.display_upper95.max()
    padding=.08*(high-low);limits=(float(low-padding),float(high+padding))
    parameter_limits[block]=limits
    fig,axes=plt.subplots(1,3,figsize=(15.5,5),sharey=True)
    for ax,abl in zip(axes,ABLS):
        for group,offset in [('wt',-.08),('hom',.08)]:
            z=table.loc[table.group.eq(group)&table.ABL.eq(abl)].sort_values('ILD')
            ax.errorbar(np.arange(10)+offset,z.display_mean,
                yerr=np.vstack([z.display_mean-z.display_lower95,z.display_upper95-z.display_mean]),
                fmt='o',color=GROUP_COLORS[group],ms=4.5,capsize=3,elinewidth=1.1,linestyle='none')
        ax.set(title=f'ABL {abl} dB',xticks=np.arange(10),xticklabels=ILDS,ylim=limits,
               xlabel='Signed ILD (dB; categorical spacing)',xlim=(-.6,9.6))
        ax.tick_params(axis='x',rotation=45)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f'SHANK3 cohort 1 | {ylabel.split(" (")[0]} | WT vs HOM',y=.98,fontsize=15)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False)
    fig.text(.5,.025,'Dots: posterior means; bars: central 95% posterior credible intervals (10,000 saved draws).\n'
        'WT: 5 animals / 68,184 choices; HOM: 6 animals / 67,715 choices. Uncertainty is conditional on frozen proactive/lapse parameters.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.25,wspace=.13)
    save_figure(fig,f'SHANK3_cohort1_WT_HOM_{block}_comparison',f'{ylabel}: three ABL panels')

fig,axes=plt.subplots(1,2,figsize=(8.4,4.8))
for ax,block,ylabel in zip(axes,['w','del_go'],['Starting bias w','Readout delay del_go (ms)']):
    z=parameters.loc[parameters.block.eq(block)].set_index('group')
    low,high=z.display_lower95.min(),z.display_upper95.max();padding=.1*(high-low)
    parameter_limits[block]=[float(low-padding),float(high+padding)]
    for i,group in enumerate(['wt','hom']):
        row=z.loc[group]
        ax.errorbar(i,row.display_mean,yerr=[[row.display_mean-row.display_lower95],[row.display_upper95-row.display_mean]],
            fmt='o',color=GROUP_COLORS[group],ms=6,capsize=4,elinewidth=1.2)
    ax.set(xticks=[0,1],xticklabels=['WT','HOM'],xlim=(-.6,1.6),ylabel=ylabel,ylim=parameter_limits[block])
fig.suptitle('SHANK3 cohort 1 | Shared reactive parameters',fontsize=14)
fig.text(.5,.035,'Posterior means and central 95% credible intervals.\nConditional variational uncertainty; not animal-to-animal variability.',ha='center',fontsize=9)
fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.22,wspace=.4)
save_figure(fig,'SHANK3_cohort1_WT_HOM_shared_parameter_comparison','Shared w and del_go')


# %% Three compact chronometrics; pooled means and SEM across animal means
def chronometric_figure(group=None):
    merged=group is None
    fig,ax=plt.subplots(figsize=(6.4,6.9))
    ax.set_box_aspect(1)
    curves=[(g,None,GROUP_COLORS[g],LABELS[g],offset) for g,offset in [('wt',-.04),('hom',.04)]] if merged else [
        (group,abl,ABL_COLORS[abl],f'ABL {abl} dB',offset) for abl,offset in zip(ABLS,[-.06,0,.06])]
    legend=[]
    for g,abl,color,label,offset in curves:
        z=points.loc[points.group.eq(g)&points.view.eq('ABL_pooled' if merged else 'ABL_separate')]
        if abl is not None:
            z=z.loc[z.ABL.eq(abl)]
        z=z.sort_values('abs_ILD');x=np.arange(5)+offset
        ax.plot(x,z.model_mean_RT_s,color=color,linewidth=MODEL_LINEWIDTH,zorder=2)
        ax.errorbar(x,z.data_mean_RT_s,yerr=z.animal_SEM_s,fmt='o',color=color,ms=5,
            capsize=3,elinewidth=1.,linestyle='none',zorder=3)
        legend.append(Line2D([],[],color=color,linewidth=MODEL_LINEWIDTH,label=label))
    legend.extend([Line2D([],[],color='.2',marker='o',linestyle='',label='Data mean ± animal SEM'),
                   Line2D([],[],color='.2',linewidth=MODEL_LINEWIDTH,label='Model mean')])
    ax.set(xticks=np.arange(5),xticklabels=ABS_ILDS,xlim=(-.35,4.35),ylim=CHRONOMETRIC_YLIM,
           xlabel='|ILD| (dB)',ylabel='Mean RT (s)')
    title='WT vs HOM | ABLs pooled' if merged else f'{LABELS[group]} | ABLs separate'
    fig.suptitle(f'SHANK3 cohort 1 | {title}',y=.97,fontsize=13)
    subtitle='WT: 5 animals / 68,184 choices; HOM: 6 animals / 67,715 choices' if merged else f"{groups[group]['animals']} animals; {groups[group]['trials']:,} valid choices"
    fig.text(.5,.915,subtitle,ha='center',fontsize=9)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.22),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.025,'Valid choices in [0,1] s. Dots: pooled-trial means; bars: SEM across animal means.\n'
        'Model: complete-mixture mean conditional on the same valid window.\nSigns and pooled ABLs use retained-trial weights; model lines are 1.0 pt.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.14,right=.97,top=.86,bottom=.32)
    stem='SHANK3_cohort1_WT_HOM_ABL_pooled_chronometric' if merged else f'SHANK3_cohort1_{LABELS[group]}_ABL_chronometric'
    save_figure(fig,stem,title)


for group in FIT_ROOTS:
    chronometric_figure(group)
chronometric_figure()
assert len(figures)==7 and len(artifacts)==14
for name,digest in baseline.items():
    assert sha256(ROOT/name)==digest,name
manifest={'status':'generated_pending_independent_verification','figures':figures,'artifacts':artifacts,
    'groups':groups,'fit_roots':{g:str(p.relative_to(ROOT)) for g,p in FIT_ROOTS.items()},
    'parameter_estimates':184,'signed_model_means':60,'chronometric_pairs':40,
    'animal_ABL_abs_means':165,'animal_ABL_pooled_abs_means':55,'posterior_draws_per_group':10000,
    'group_colors':GROUP_COLORS,'ABL_colors':ABL_COLORS,'model_linewidth':MODEL_LINEWIDTH,
    'chronometric_ylim':CHRONOMETRIC_YLIM,'chronometric_panel_box_aspect':1.,
    'parameter_ylim':parameter_limits,'quadrature_orders':QUADRATURE_ORDERS,
    'max_density_vs_saved_CDF_mean_error_s':float(np.abs(signed.model_mean_RT_s-signed.model_mean_from_saved_CDF_s).max()),
    'posterior_intervals':'central 95% quantiles; conditional variational uncertainty',
    'chronometric_uncertainty':'sample SD across animal means / sqrt(animal count), around pooled-trial dots',
    'pooling':'actual fitted valid-trial counts; each onset normalized before model averaging',
    'input_and_fit_hashes_unchanged':True,'source':str(Path(__file__).relative_to(ROOT)),
    'source_sha256':sha256(Path(__file__))}
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved seven PNGs/PDFs and all numerical audits:',OUT,flush=True)
