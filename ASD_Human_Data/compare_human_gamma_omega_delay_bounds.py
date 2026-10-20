# %%
"""Audit 300 ms pressure, then compare identical-trial 300/500 ms SVI fits."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/human_gamma_omega_delay_comparison_mpl')
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% Only the delay hard upper limit changes; keep flat limits and initialization.
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir',type=Path,default=ROOT/'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay300ms')
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay500ms')
parser.add_argument('--audit-only',action='store_true')
args=parser.parse_args();old=args.baseline_dir;out=args.output_dir
configs=[json.loads((p/'run_config.json').read_text()) for p in [old,out]]
assert configs[0]['priors']['t_E_aff']['hard']==[0,.3]
assert configs[1]['priors']['t_E_aff']['hard']==[0,.5]
for key in configs[0]:
    if key not in ['priors','source_hashes']:assert configs[0][key]==configs[1][key],key
for key in configs[0]['priors']:
    if key!='t_E_aff':assert configs[0]['priors'][key]==configs[1]['priors'][key]
assert configs[0]['priors']['t_E_aff']['plausible']==configs[1]['priors']['t_E_aff']['plausible']==[.01,.2]
assert (old/'selected_trials.csv').read_bytes()==(out/'selected_trials.csv').read_bytes()
assert (old/'excluded_trials.csv').read_bytes()==(out/'excluded_trials.csv').read_bytes()
rows=[];report={}
for group in ['Control','ASD']:
    a=old/group.lower();b=out/group.lower()
    assert json.loads((a/'initialization.json').read_text())==json.loads((b/'initialization.json').read_text())
    c=pd.read_csv(a/'conditions.csv',float_precision='round_trip')
    samples=np.load(a/'posterior_samples.npz')['t_E_aff']
    for r in c.itertuples():
        x=samples[:,r.condition_index]
        rows.append(dict(Group=group,condition_index=r.condition_index,ASL_cd=r.ASL_cd,signed_delta=r.signed_delta,
            mean_ms=x.mean()*1000,q025_ms=np.quantile(x,.025)*1000,q975_ms=np.quantile(x,.975)*1000,
            P_delay_gt_290ms=float(np.mean(x>.29)),P_delay_gt_295ms=float(np.mean(x>.295)),
            P_delay_gt_299ms=float(np.mean(x>.299)),minimum_RT_ms=r.minimum_RT_s*1000,
            old_causal_limit_ms=r.guide_delay_upper_s*1000,
            new_causal_limit_ms=min(.5,r.minimum_RT_s-1e-6)*1000))
audit=pd.DataFrame(rows)
for group,g in audit.groupby('Group',sort=False):
    report[group]=dict(conditions=20,upper_95pct_endpoint_at_least_295ms=int(g.q975_ms.ge(295).sum()),
        conditions_with_at_least_10pct_mass_above_290ms=int(g.P_delay_gt_290ms.ge(.1).sum()),
        largest_posterior_mean_ms=float(g.mean_ms.max()),largest_95pct_endpoint_ms=float(g.q975_ms.max()),
        conditions_gaining_causal_support=int((g.new_causal_limit_ms-g.old_causal_limit_ms).gt(.001).sum()),
        largest_new_causal_limit_ms=float(g.new_causal_limit_ms.max()))
audit.to_csv(out/'delay300_boundary_audit.csv',index=False,float_format='%.17g')
(out/'delay300_boundary_diagnosis.json').write_text(json.dumps(dict(
    groups=report,only_analysis_change='Delay hard upper prior 0.3 -> 0.5 s; flat 0.01--0.2 s unchanged',
    same_trials=True,same_exclusions=True,same_initial_physical_values=True,same_other_fit_settings=True,
    causality='No delay can exceed its own condition minimum RT; no retained trials removed',
    baseline_config_sha256=hashlib.sha256((old/'run_config.json').read_bytes()).hexdigest(),
    refit_config_sha256=hashlib.sha256((out/'run_config.json').read_bytes()).hexdigest()),indent=2))
print(json.dumps(report,indent=2))
if args.audit_only:sys.exit(0)

# %% Join every parameter summary and assess same-row fit quality without ELBO comparison.
before=pd.read_csv(old/'parameter_summary.csv',float_precision='round_trip')
after=pd.read_csv(out/'parameter_summary.csv',float_precision='round_trip')
assert before[['Group','parameter']].equals(after[['Group','parameter']])
keys=['Group','parameter','block','condition_index','ASL_cd','signed_delta']
comp=before.merge(after,on=keys,suffixes=('_300','_500'),validate='one_to_one')
comp['change_in_posterior_mean']=comp.mean_500-comp.mean_300
comp.to_csv(out/'delay_bound_parameter_comparison.csv',index=False,float_format='%.17g')
scores=[]
for directory,limit in [(old,300),(out,500)]:
    q=pd.read_csv(directory/'fit_quality.csv')
    q['delay_upper_prior_ms']=limit
    for i,row in q.iterrows():
        r=json.loads((directory/row.Group.lower()/'convergence_summary.json').read_text())
        q.loc[i,'completed_steps']=r['completed_steps'];q.loc[i,'restored_best_step']=r['restored_best_step']
        q.loc[i,'convergence_flagged']=r['flagged']
    scores.append(q)
scores=pd.concat(scores,ignore_index=True)
scores.to_csv(out/'delay_bound_fit_comparison.csv',index=False,float_format='%.17g')
delay=comp.loc[comp.block.eq('t_E_aff')].copy()
delay=delay.merge(audit,on=['Group','condition_index','ASL_cd','signed_delta'],validate='one_to_one')
for group in ['Control','ASD']:
    p=np.load(out/group.lower()/'posterior_samples.npz')['t_E_aff']
    for index,row in delay.loc[delay.Group.eq(group)].iterrows():
        delay.loc[index,'P_new_delay_gt_300ms']=np.mean(p[:,int(row.condition_index)]>.3)
delay.to_csv(out/'delay_bound_delay_comparison.csv',index=False,float_format='%.17g')

# %% Two group rows: delay comparison at each ASL plus the pooled RTD comparison.
selected=pd.read_csv(out/'selected_trials.csv',float_precision='round_trip')
curves={limit:pd.read_csv(directory/'rtd_model_curves.csv',float_precision='round_trip') for directory,limit in [(old,300),(out,500)]}
colors={'Control':'#1976d2','ASD':'#d62728'};asl_colors={20:'#1f77b4',40:'#ff7f0e'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(18.5,10.2))
grid=fig.add_gridspec(2,3,width_ratios=[1,1,1.2],left=.06,right=.985,bottom=.14,top=.84,hspace=.36,wspace=.27)
pooled_curves=[];pooled_histograms=[];rtd_axes=[];rtd_ymax=0.
for r,group in enumerate(['Control','ASD']):
    for col,level in enumerate([20,40]):
        ax=fig.add_subplot(grid[r,col])
        p=delay.loc[delay.Group.eq(group)&delay.ASL_cd.eq(level)].sort_values('signed_delta')
        for limit,shift,color,marker in [(300,-.018,'.55','o'),(500,.018,colors[group],'o' if group=='Control' else '*')]:
            y=p[f'mean_{limit}'].to_numpy()*1000
            lo=p[f'q025_{limit}'].to_numpy()*1000;hi=p[f'q975_{limit}'].to_numpy()*1000
            ax.errorbar(p.signed_delta+shift,y,yerr=np.vstack([y-lo,hi-y]),fmt=marker,ms=5 if marker=='o' else 8,
                        color=color,capsize=2,lw=1.1,mfc='white' if limit==300 else color)
        ax.scatter(p.signed_delta,p.new_causal_limit_ms,marker='_',color='.25',s=65,alpha=.6)
        ax.axhline(300,color='.55',ls=':',lw=1)
        ax.set(title=f'{group} | ASL {level} cd/m²',ylabel='Delay (ms)' if col==0 else '',
               ylim=(0,505),xlim=(-2.14,2.14),xticks=p.signed_delta)
        ax.tick_params(axis='x',labelsize=8,rotation=55);ax.grid(alpha=.18)
        if r==1:ax.set_xlabel('Signed ΔdB (right − left)')
    ax=fig.add_subplot(grid[r,2])
    rtd_axes.append(ax)
    for level in [20,40]:
        obs=selected.loc[selected.Group.eq(group)&selected.ASL_cd.eq(level)]
        edges=np.linspace(0,1.8,91);hist,_=np.histogram(obs.RT_s,bins=edges)
        rtd_ymax=max(rtd_ymax,float(np.max(hist/(len(obs)*.02))))
        pooled_histograms.append(pd.DataFrame(dict(Group=group,ASL_cd=level,left_s=edges[:-1],right_s=edges[1:],
            n=len(obs),count=hist,density=hist/(len(obs)*.02))))
        ax.stairs(hist/(len(obs)*.02),edges,color=asl_colors[level],alpha=.35,lw=1)
        for limit,style in [(300,'--'),(500,'-')]:
            table=curves[limit].loc[curves[limit].Group.eq(group)&curves[limit].ASL_cd.eq(level)]
            combined=np.zeros(1801)
            for delta in [.1,.3,.65,1.3,2.]:
                c=table.loc[table.abs_delta.eq(delta)]
                n=int(np.isclose(obs.signed_delta.abs(),delta).sum())
                combined+=c.density.to_numpy()*n/len(obs)
            ax.plot(c.time_s,combined,color=asl_colors[level],ls=style,lw=1.8)
            rtd_ymax=max(rtd_ymax,float(combined.max()))
            pooled_curves.append(pd.DataFrame(dict(Group=group,ASL_cd=level,delay_upper_prior_ms=limit,
                time_s=c.time_s.to_numpy(),density=combined)))
    ax.set(title=f'{group} | pooled RTD',xlim=(0,1.8),xlabel='RT (s)',ylabel='Density (s⁻¹)')
    ax.grid(alpha=.18)
for ax in rtd_axes:ax.set_ylim(0,np.ceil(rtd_ymax*1.08*2)/2)
pd.concat(pooled_curves,ignore_index=True).to_csv(out/'delay_bound_pooled_rtd_model_curves.csv',index=False,float_format='%.17g')
pd.concat(pooled_histograms,ignore_index=True).to_csv(out/'delay_bound_pooled_rtd_histograms.csv',index=False,float_format='%.17g')
fig.suptitle('Human RT | delay-prior sensitivity: 300 → 500 ms',fontsize=21,y=.97)
fig.text(.5,.925,'Same trials, likelihood, priors for all other parameters, initialization, seed and SVI stopping rule',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='.55',marker='o',mfc='white',ls='none',label='300 ms prior'),
    Line2D([],[],color='#1976d2',marker='o',ls='none',label='500 ms prior: Control'),
    Line2D([],[],color='#d62728',marker='*',ls='none',label='500 ms prior: ASD'),
    Line2D([],[],color='.25',marker='_',ls='none',label='Condition causal limit'),
    Line2D([],[],color='.2',ls='--',label='RTD: 300 ms prior'),Line2D([],[],color='.2',ls='-',label='RTD: 500 ms prior')],
    loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.065,'Delay bars: equal-tailed 95% variational credible intervals. Dotted line: old 300 ms cap. RTD blue/orange: ASL 20/40.',ha='center',fontsize=10)
fig.text(.5,.032,'Condition = RT; raw-trial pooling; 0–1.8 s retained; both choices. Each signed-condition density is normalized before mixing.',ha='center',fontsize=10)
for suffix in ['png','pdf']:fig.savefig(out/f'human_gamma_omega_delay_prior_comparison.{suffix}',dpi=180)
plt.close(fig)
(out/'delay_bound_comparison_manifest.json').write_text(json.dumps(dict(
    figures=['human_gamma_omega_delay_prior_comparison.png','human_gamma_omega_delay_prior_comparison.pdf'],
    tables=['delay300_boundary_audit.csv','delay_bound_parameter_comparison.csv','delay_bound_fit_comparison.csv','delay_bound_delay_comparison.csv',
            'delay_bound_pooled_rtd_model_curves.csv','delay_bound_pooled_rtd_histograms.csv'],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(scores.to_string(index=False))
