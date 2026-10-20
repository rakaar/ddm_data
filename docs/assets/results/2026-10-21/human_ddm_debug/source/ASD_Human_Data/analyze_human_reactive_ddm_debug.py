# %%
"""Compare same-row human TIED/DDM diagnostics, including posterior prediction."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/human_ddm_debug_mpl')
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from human_reactive_tied_rt_only_utils import ROOT,PARAM_NAMES,load_trials,group_data,sha256
from human_plain_ddm_diagnostics_utils import (
    ddm_log_density,ddm_log_cdf,tied_coordinates,physical_to_tied,scipy_ddm,
)

# %% Read-only sources, 2-ms diagnostic curves and the original 20-ms data bins.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
parser.add_argument('--posterior-draws',type=int,default=128)
args=parser.parse_args();out=args.output_dir
protocol=json.loads((out/'protocol.json').read_text());baseline=Path(protocol['baseline_dir'])
frame,provenance=load_trials(0.,1.8)
times=np.linspace(0.,1.8,901);edges=np.linspace(0.,1.8,91)
colors={20:'#1f77b4',40:'#ff7f0e'}
styles={
    'SVI_mean':('#777777','--','TIED: SVI mean'),
    'SVI_predictive':('#202020',':','TIED: posterior predictive'),
    'TIED_RT':('#9467bd','-.','TIED: RT-only MLE'),
    'DDM_shared_RT':('#008b8b','-','DDM: shared scale, RT only'),
    'DDM_free_RT':('#2ca02c','-','DDM: free scales, RT only'),
    'TIED_joint':('#9467bd','-.','TIED: RT + choice'),
    'DDM_shared_joint':('#008b8b','-','DDM: shared scale, RT + choice'),
    'DDM_free_joint':('#2ca02c','-','DDM: free scales, RT + choice'),
}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
payload={};quality=[];condition_rows=[];parameter_rows=[];hist_rows=[];posterior_checks=[]


def cdf_gap(ordered,predicted):
    n=len(ordered)
    return float(max(np.max(np.arange(1,n+1)/n-predicted),np.max(predicted-np.arange(n)/n)))


@jax.jit
def curves(gamma,omega,w,delay):
    g,o=gamma[:,None],omega[:,None]
    logmass=ddm_log_cdf(1.8,g,o,w,delay)
    f=jnp.exp(ddm_log_density(jnp.asarray(times)[None,:],g,o,w,delay)-logmass)
    F=jnp.exp(ddm_log_cdf(jnp.asarray(times)[None,:],g,o,w,delay)-logmass)
    return f,F,jnp.exp(logmass[:,0])


# %% Every comparison keeps the signed-condition frequencies and retained rows.
for group in ['Control','ASD']:
    folder=out/group.lower();rtfit=json.loads((folder/'fits.json').read_text())
    joint=json.loads((folder/'joint_fits.json').read_text())
    shared=json.loads((folder/'shared_scale_fits.json').read_text())
    sub,conditions,data,_=group_data(frame,group)
    B,D=data['condition_B'],data['condition_D'];reference=rtfit['reference_B']
    rt,choice,ids=data['rt'],data['choice'],data['condition']
    samples=np.load(baseline/group.lower()/'posterior_samples.npz')
    means=np.array([samples[name].mean() for name in PARAM_NAMES])
    specs={}
    for label,x in [('SVI_mean',physical_to_tied(means,reference)),('TIED_RT',rtfit['tied']['x']),
                    ('TIED_joint',joint['tied']['x'])]:
        g,o,w,delay=tied_coordinates(jnp.asarray(x),B,D,reference)
        specs[label]=(np.asarray(g),np.asarray(o),float(w),float(delay),5)
    for label,result,mode in [('DDM_free_RT',rtfit['free_ddm'],'rt'),('DDM_free_joint',joint['free_ddm'],'joint')]:
        x=np.array(result['x'])
        g=np.sign(np.asarray(D))*np.exp(x[:20]) if mode=='rt' else np.sinh(x[:20])
        specs[label]=(g,np.exp(x[20:40]),x[40],x[41],42)
    for label,key in [('DDM_shared_RT','rt_only'),('DDM_shared_joint','joint')]:
        x=np.array(shared[key]['x'])
        specs[label]=(np.sinh(x[:20]),np.full(20,np.exp(x[20])),x[21],x[22],23)
    payload[group]=dict(sub=sub,conditions=conditions,models={})
    for label,(g,o,w,delay,nparams) in specs.items():
        f,F,mass=[np.asarray(a) for a in curves(jnp.asarray(g),jnp.asarray(o),w,delay)]
        right=scipy_ddm(1.8,g,o,w,delay,cumulative=True,side=1)/mass
        llrt=float(jnp.sum(ddm_log_density(rt,jnp.asarray(g)[ids],jnp.asarray(o)[ids],w,delay)
                           -jnp.log(jnp.asarray(mass)[ids])))
        lljoint=float(jnp.sum(ddm_log_density(rt,jnp.asarray(g)[ids],jnp.asarray(o)[ids],w,delay,side=choice)
                              -jnp.log(jnp.asarray(mass)[ids])))
        payload[group]['models'][label]=dict(pdf=f,cdf=F,right=right,gamma=g,omega=o,w=w,delay=delay,
                                            n_parameters=nparams,RT_LL=llrt,joint_LL=lljoint)
        for row in conditions.itertuples():
            i=row.condition_index
            parameter_rows.append(dict(Group=group,model=label,ASL_cd=row.ASL_cd,Delta_dB=row.signed_delta,
                                       gamma=g[i],omega=o[i],w=w,delay_s=delay,n_parameters=nparams,
                                       empirical_P_right=row.right/row.n,predicted_P_right=right[i]))
        print(group,label,'RT LL',llrt,'joint LL',lljoint,'w,delay',w,delay,flush=True)

    # Average predictions over complete joint variational draws. This preserves
    # posterior correlations instead of substituting five marginal means.
    rng=np.random.default_rng(0);indices=rng.choice(len(samples[PARAM_NAMES[0]]),args.posterior_draws,replace=False)
    fs=[];Fs=[];rights=[]
    for index in indices:
        p=np.array([samples[name][index] for name in PARAM_NAMES])
        g,o,w,delay=tied_coordinates(jnp.asarray(physical_to_tied(p,reference)),B,D,reference)
        f,F,mass=[np.asarray(a) for a in curves(g,o,w,delay)]
        fs.append(f);Fs.append(F)
        rights.append(scipy_ddm(1.8,np.asarray(g),np.asarray(o),float(w),float(delay),cumulative=True,side=1)/mass)
    f=np.mean(fs,axis=0);F=np.mean(Fs,axis=0)
    payload[group]['models']['SVI_predictive']=dict(pdf=f,cdf=F,right=np.mean(rights,axis=0),n_parameters=5,
                                                  RT_LL=np.nan,joint_LL=np.nan)
    np.savez_compressed(folder/'posterior_predictive.npz',time_s=times,pdf=f,cdf=F,
                        pdf_q025=np.quantile(fs,.025,axis=0),pdf_q975=np.quantile(fs,.975,axis=0),
                        sample_indices=indices)
    posterior_checks.append(dict(Group=group,n_draws=len(indices),
        max_condition_cdf_change=float(np.max(abs(F-payload[group]['models']['SVI_mean']['cdf'])))))

    for label,model in payload[group]['models'].items():
        f,F=model['pdf'],model['cdf'];gaps=[]
        for row in conditions.itertuples():
            i=row.condition_index;selected=sub.loc[sub.condition_index.eq(i)]
            ordered=np.sort(selected.RT_s.to_numpy())
            prediction=np.interp(ordered,times,F[i])
            gap=cdf_gap(ordered,prediction);gaps.append(gap)
            model_mean=np.trapz(times*f[i],times)
            model_sd=np.sqrt(np.trapz((times-model_mean)**2*f[i],times))
            condition_rows.append(dict(Group=group,model=label,ASL_cd=row.ASL_cd,Delta_dB=row.signed_delta,
                n=row.n,cdf_max_error=gap,data_mean_s=ordered.mean(),data_sd_s=ordered.std(ddof=1),
                model_mean_s=model_mean,model_sd_s=model_sd,empirical_P_right=row.right/row.n,
                predicted_P_right=model['right'][i]))
        model['mixtures']={};pooled_gaps=[]
        for asl in [20,40]:
            for delta in [None,.1,.3,.65,1.3,2.]:
                mask=conditions.ASL_cd.eq(asl);trial_mask=sub.ASL_cd.eq(asl)
                if delta is not None:
                    mask &= conditions.signed_delta.abs().eq(delta)
                    trial_mask &= sub.signed_delta.abs().eq(delta)
                indices=conditions.loc[mask,'condition_index'].to_numpy()
                selected=sub.loc[trial_mask];weights=conditions.loc[mask,'n'].to_numpy()/len(selected)
                pdf=np.average(f[indices],axis=0,weights=weights);cdf=np.average(F[indices],axis=0,weights=weights)
                model['mixtures'][(asl,delta)]=(pdf,cdf)
                if label=='SVI_mean':
                    counts=np.histogram(selected.RT_s,edges)[0]
                    hist_rows.extend(dict(Group=group,ASL_cd=asl,abs_Delta_dB='all' if delta is None else delta,
                        bin_left_s=edges[i],bin_right_s=edges[i+1],n=len(selected),count=int(count),density=count/len(selected)/.02)
                        for i,count in enumerate(counts))
                if delta is None:
                    ordered=np.sort(selected.RT_s.to_numpy())
                    pooled_gaps.append(cdf_gap(ordered,np.interp(ordered,times,cdf)))
        empirical=conditions.right.to_numpy()/conditions.n.to_numpy()
        quality.append(dict(Group=group,model=label,n_trials=len(sub),n_parameters=model['n_parameters'],
            RT_log_likelihood=model['RT_LL'],joint_log_likelihood=model['joint_LL'],
            mean_condition_cdf_error=np.mean(gaps),max_condition_cdf_error=max(gaps),
            mean_pooled_asl_cdf_error=np.mean(pooled_gaps),psychometric_RMSE=np.sqrt(np.mean((model['right']-empirical)**2))))
    np.savez_compressed(folder/'comparison_curves.npz',time_s=times,
        **{label+'_'+key:model[key] for label,model in payload[group]['models'].items() for key in ['pdf','cdf','right']})

pd.DataFrame(quality).to_csv(out/'fit_quality.csv',index=False,float_format='%.17g')
pd.DataFrame(condition_rows).to_csv(out/'condition_fit_quality.csv',index=False,float_format='%.17g')
pd.DataFrame(parameter_rows).to_csv(out/'condition_parameters.csv',index=False,float_format='%.17g')
pd.DataFrame(hist_rows).to_csv(out/'rtd_histograms.csv',index=False,float_format='%.17g')
pd.DataFrame(posterior_checks).to_csv(out/'posterior_predictive_check.csv',index=False,float_format='%.17g')

# %% Pooled RTD comparisons and a separate posterior-predictive artifact check.
footer=('Condition = RT | 0–1.8 s; all recorded choices retained | delay bound 0–300 ms; causal support below earliest group RT\n'
        'Raw-trial pooling; actual signed-condition frequencies | densities normalized on 0–1.8 s | MLE diagnostics use no priors')
for kind,labels,title in [
    ('rt_only',['SVI_mean','TIED_RT','DDM_shared_RT','DDM_free_RT'],'RT-only likelihood: TIED versus plain DDM'),
    ('joint',['TIED_joint','DDM_shared_joint','DDM_free_joint'],'Joint RT + choice likelihood: TIED versus plain DDM'),
    ('svi_predictive',['SVI_mean','SVI_predictive'],'SVI diagnostic: physical mean parameters versus posterior prediction')]:
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True)
    for ri,group in enumerate(['Control','ASD']):
        for ci,asl in enumerate([20,40]):
            ax=axes[ri,ci];sub=payload[group]['sub'];rts=sub.loc[sub.ASL_cd.eq(asl),'RT_s']
            heights=np.histogram(rts,edges)[0]/len(rts)/.02
            ax.stairs(heights,edges,color=colors[asl],lw=1.6,label='Data (20 ms bins)')
            for label in labels:
                color,ls,text=styles[label];pdf=payload[group]['models'][label]['mixtures'][(asl,None)][0]
                ax.plot(times,pdf,color=color,ls=ls,lw=1.8,label=text)
            ax.set_title(f'{group} | ASL {asl} cd/m² | n={len(rts):,}')
            ax.set_xlim(0,1.8);ax.set_xticks([0,.5,1,1.5,1.8]);ax.grid(alpha=.16)
            if ci==0:ax.set_ylabel('RT density (s⁻¹)')
            if ri==1:ax.set_xlabel('Stored RT_s (s)')
    handles,labels_legend=axes[0,0].get_legend_handles_labels()
    fig.suptitle(title,fontsize=18,y=.975)
    fig.legend(handles,labels_legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.5,.025,footer,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.88))
    for ext in ['png','pdf']:fig.savefig(out/f'human_{kind}_ddm_debug.{ext}',dpi=180)
    plt.close(fig)

# %% RTDs by absolute difference: common-scale plain DDM and TIED, both RT only.
fig,axes=plt.subplots(2,5,figsize=(20,8),sharex=True,sharey=True)
for ri,group in enumerate(['Control','ASD']):
    sub=payload[group]['sub']
    for ci,delta in enumerate([.1,.3,.65,1.3,2.]):
        ax=axes[ri,ci]
        for asl in [20,40]:
            rts=sub.loc[sub.ASL_cd.eq(asl)&sub.signed_delta.abs().eq(delta),'RT_s']
            ax.stairs(np.histogram(rts,edges)[0]/len(rts)/.02,edges,color=colors[asl],lw=1.1,alpha=.75)
            for label,ls in [('DDM_shared_RT','-'),('TIED_RT','--')]:
                pdf=payload[group]['models'][label]['mixtures'][(asl,delta)][0]
                ax.plot(times,pdf,color=colors[asl],ls=ls,lw=1.6)
        ax.set_title(f'{group} | |ΔdB| = {delta:g}')
        ax.set_xlim(0,1.8);ax.set_xticks([0,.5,1,1.5]);ax.grid(alpha=.15)
        if ci==0:ax.set_ylabel('RT density (s⁻¹)')
        if ri==1:ax.set_xlabel('RT (s)')
legend=[Line2D([],[],color=colors[a],label=f'ASL {a} cd/m²') for a in [20,40]]
legend += [Line2D([],[],color='.2',ls='-',label='Plain DDM: shared scale, RT only'),
           Line2D([],[],color='.2',ls='--',label='TIED: RT-only MLE')]
fig.suptitle('Same trials and likelihood: free stimulus drifts versus TIED stimulus mapping',fontsize=18,y=.975)
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
fig.text(.5,.025,footer,ha='center',fontsize=9);fig.tight_layout(rect=(0,.08,1,.9))
for ext in ['png','pdf']:fig.savefig(out/f'human_ddm_rt_by_delta_debug.{ext}',dpi=180)
plt.close(fig)

# %% Choices are a diagnostic prediction, including for models that did not fit them.
fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
for ri,group in enumerate(['Control','ASD']):
    cond=payload[group]['conditions']
    for ci,asl in enumerate([20,40]):
        ax=axes[ri,ci];mask=cond.ASL_cd.eq(asl);x=cond.loc[mask,'signed_delta']
        p=cond.loc[mask,'right']/cond.loc[mask,'n']
        ax.errorbar(x,p,yerr=np.sqrt(p*(1-p)/cond.loc[mask,'n']),fmt='o',color=colors[asl],label='Data ± binomial SE')
        for label in ['DDM_shared_RT','TIED_joint','DDM_shared_joint','DDM_free_joint']:
            color,ls,text=styles[label]
            if label=='DDM_shared_RT':ls=':'
            ax.plot(x,payload[group]['models'][label]['right'][mask],color=color,ls=ls,label=text,lw=1.8)
        ax.set_title(f'{group} | ASL {asl} cd/m²');ax.set_ylim(-.03,1.03);ax.grid(alpha=.16)
        if ci==0:ax.set_ylabel('P(right), conditional on retained RT')
        if ri==1:ax.set_xlabel('Signed ΔdB (right − left)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.suptitle('Does fitting RT shape alone imply plausible choices?',fontsize=18,y=.975)
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
fig.text(.5,.023,'RT-only model did not use observed choices; drift direction follows the stimulus sign for interpretation.\n'+footer,ha='center',fontsize=9)
fig.tight_layout(rect=(0,.095,1,.87))
for ext in ['png','pdf']:fig.savefig(out/f'human_ddm_choice_debug.{ext}',dpi=180)
plt.close(fig)

# %% Mean absolute-Gamma/Omega across ±D explains which constraints were relaxed.
parameters=pd.DataFrame(parameter_rows)
fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
for ri,group in enumerate(['Control','ASD']):
    for ci,param in enumerate(['gamma','omega']):
        ax=axes[ri,ci]
        for asl in [20,40]:
            for label,ls,marker in [('TIED_RT','--',None),('DDM_shared_RT','-','o'),('DDM_free_RT',':','s')]:
                select=parameters.loc[parameters.Group.eq(group)&parameters.model.eq(label)&parameters.ASL_cd.eq(asl)].copy()
                select['abs_delta']=select.Delta_dB.abs();select['gamma']=select.gamma.abs()
                mean=select.groupby('abs_delta')[param].mean()
                ax.plot(mean.index,mean.values,color=colors[asl],ls=ls,marker=marker,lw=1.7)
        ax.set_title(f'{group} | '+('Drift–bound product |Γ|' if param=='gamma' else 'Time scale Ω (s⁻¹)'))
        ax.grid(alpha=.16);ax.set_ylabel('|Γ|' if param=='gamma' else 'Ω (s⁻¹)')
        if ri==1:ax.set_xlabel('|ΔdB|')
legend=[Line2D([],[],color=colors[a],label=f'ASL {a} cd/m²') for a in [20,40]]
legend += [Line2D([],[],color='.3',ls='--',label='TIED: RT-only MLE'),
           Line2D([],[],color='.3',marker='o',ls='-',label='Plain DDM: shared scale'),
           Line2D([],[],color='.3',marker='s',ls=':',label='Plain DDM: free scales')]
fig.suptitle('RT shape needs strong drift even at the smallest stimulus differences',fontsize=18,y=.975)
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
fig.text(.5,.025,'Each point averages derived parameters across the two signed conditions; these are not fitted group means.\n'+footer,ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.87))
for ext in ['png','pdf']:fig.savefig(out/f'human_ddm_parameter_constraints_debug.{ext}',dpi=180)
plt.close(fig)

# %% Transparent artifacts and old-fit preservation.
sources=['human_plain_ddm_diagnostics_utils.py','debug_human_reactive_ddm.py','debug_human_ddm_joint.py',
         'debug_human_ddm_shared_scale.py','analyze_human_reactive_ddm_debug.py']
manifest=dict(source_hashes={f'ASD_Human_Data/{name}':sha256(ROOT/'ASD_Human_Data'/name) for name in sources},
              data=provenance,grid_step_s=.002,histogram_bin_s=.02,posterior_predictive_draws=args.posterior_draws,
              figures={p.name:sha256(p) for p in out.glob('*.png')},fit_quality='fit_quality.csv')
(out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for path,digest in json.loads((out/'baseline_sha256.json').read_text()).items():assert sha256(ROOT/path)==digest,path
print(pd.DataFrame(quality).to_string(index=False),flush=True)
print('All figures, metrics and baseline-preservation checks completed.',flush=True)
