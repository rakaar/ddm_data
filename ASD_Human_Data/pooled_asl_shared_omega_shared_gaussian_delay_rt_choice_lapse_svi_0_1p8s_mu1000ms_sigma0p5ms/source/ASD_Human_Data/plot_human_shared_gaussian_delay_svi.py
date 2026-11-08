# %%
"""Control/ASD RTDs, psychometrics and Gaussian-delay posterior comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/human_gaussian_delay_mpl')
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax
import jax.numpy as jnp
from human_shared_gaussian_delay_svi_utils import *

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
config=json.loads((out/'run_config.json').read_text())
for path,digest in config['source_hashes'].items():assert sha256(ROOT/path)==digest,path
_,selected,_,_=load_data()
groups=['Control','ASD'];levels=[20.,40.];absolute=[.1,.3,.65,1.3,2.]
times=np.linspace(0,1.8,3601);edges=np.linspace(0,1.8,91)
level_colors={20.:'#1f77b4',40.:'#ff7f0e'}
group_colors={'Control':'#1976d2','ASD':'#d62728'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
loaded={};tables={};summaries=[];quality=[];condition_rows=[];curve_rows=[];hists=[];mixtures=[]
boundaries=[];correlations=[];derived=[]

# %% Reconstruct every plotted quantity from saved constrained posterior draws.
for group in groups:
    folder=out/group.lower()
    posterior=dict(np.load(folder/'posterior_samples.npz'))
    matrix=sample_matrix(posterior)
    assert matrix.shape==(10000,16)
    params={k:jnp.asarray(v) for k,v in vector_params(matrix.mean(0)).items()}
    sub,ct,data,_,_=prepare_group(selected,group)
    summary=pd.read_csv(folder/'posterior_summary.csv')
    np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=1e-12)
    summary.insert(0,'Group',group)
    summary['block']=[n for n in PARAMETER_BLOCKS for _ in (range(10) if n in CONDITION_BLOCKS else [0])]
    summary['condition_index']=[i for n in PARAMETER_BLOCKS for i in (range(10) if n in CONDITION_BLOCKS else [np.nan])]
    summary=summary.merge(ct[['condition_index','signed_delta']],on='condition_index',how='left',validate='many_to_one')
    summaries.append(summary)
    prediction=prediction_components(params,times)
    rows=[]
    for c in ct.itertuples():
        i=c.condition_index;rt=np.sort(sub.loc[sub.condition_index.eq(i),'RT_s'].to_numpy())
        model_cdf=np.interp(rt,times,prediction['cdf'][:,i])
        empirical=np.arange(1,len(rt)+1)/len(rt)
        gap=float(max(abs(model_cdf-empirical).max(),abs(model_cdf-empirical+1/len(rt)).max()))
        rows.append(dict(Group=group,condition_index=i,signed_delta=c.signed_delta,
            n=c.n,right=c.right,empirical_p_right=c.right/c.n,model_p_right=prediction['p_right'][i],
            cdf_gap=gap,retained_mass=prediction['retained_mass'][i]))
        curve_rows.append(pd.DataFrame(dict(Group=group,condition_index=i,
            signed_delta=c.signed_delta,time_s=times,**{k:prediction[k][:,i]
            for k in ['left_density','right_density','density','cdf']})))
    cond=pd.DataFrame(rows);condition_rows.append(cond)
    joint=float(jax.jit(lambda p:log_likelihood(p,data))(params))
    marginal=float(jax.jit(lambda p:log_likelihood(p,data,marginal=True))(params))
    errors=cond.model_p_right-cond.empirical_p_right
    quality.append(dict(Group=group,model='ASL-pooled Gamma + shared Gaussian delay/Omega + uniform lapse',free_parameters=16,
        n_trials=len(sub),joint_log_likelihood=joint,marginal_rt_log_likelihood=marginal,
        psychometric_mae=float(abs(errors).mean()),psychometric_rmse=float(np.sqrt(np.mean(errors**2))),
        mean_condition_cdf_gap=float(cond.cdf_gap.mean()),max_condition_cdf_gap=float(cond.cdf_gap.max())))
    for i,name in enumerate(PARAMETER_NAMES):
        key=summary.block.iloc[i]
        if key=='gamma':key='gamma_positive' if ct.signed_delta.iloc[i]>0 else 'gamma_negative'
        lo,hi=PRIORS[key]['hard']
        boundaries.append(dict(Group=group,parameter=name,hard_low=lo,hard_high=hi,q005=float(np.quantile(matrix[:,i],.005)),q995=float(np.quantile(matrix[:,i],.995)),
            lower_tail_distance_fraction=float((np.quantile(matrix[:,i],.005)-lo)/(hi-lo)),
            upper_tail_distance_fraction=float((hi-np.quantile(matrix[:,i],.995))/(hi-lo)),
            fraction_within_1pct_lower=float(np.mean(matrix[:,i]<lo+.01*(hi-lo))),
            fraction_within_1pct_upper=float(np.mean(matrix[:,i]>hi-.01*(hi-lo)))))
    corr=np.corrcoef(matrix,rowvar=False);ia,ib=np.triu_indices(16,1)
    for j in np.argsort(-abs(corr[ia,ib]))[:20]:
        correlations.append(dict(Group=group,parameter_1=PARAMETER_NAMES[ia[j]],parameter_2=PARAMETER_NAMES[ib[j]],correlation=corr[ia[j],ib[j]]))
    a=posterior['delay_mu']/posterior['delay_sigma']
    ratio=np.exp(-.5*a*a)/(np.sqrt(2*np.pi)*ndtr(a))
    actual_mean=posterior['delay_mu']+posterior['delay_sigma']*ratio
    actual_sd=posterior['delay_sigma']*np.sqrt(1-a*ratio-ratio*ratio)
    for label,values in [('actual_delay_mean',actual_mean),('actual_delay_sd',actual_sd)]:
        derived.append(dict(Group=group,quantity=label,mean_s=values.mean(),posterior_sd_s=values.std(ddof=1),
            q025_s=np.quantile(values,.025),q975_s=np.quantile(values,.975)))
    loaded[group]=dict(summary=summary,conditions=ct,sub=sub,fit=cond,posterior=posterior,corr=corr,
        report=json.loads((folder/'convergence_summary.json').read_text()))

tables={'parameter_summary.csv':pd.concat(summaries,ignore_index=True),
        'fit_quality.csv':pd.DataFrame(quality),'condition_fit_quality.csv':pd.concat(condition_rows,ignore_index=True),
        'psychometric_comparison.csv':pd.concat(condition_rows,ignore_index=True),
        'condition_model_curves.csv':pd.concat(curve_rows,ignore_index=True),
        'posterior_boundary_audit.csv':pd.DataFrame(boundaries),'strongest_correlations.csv':pd.DataFrame(correlations),
        'actual_truncated_delay_moments.csv':pd.DataFrame(derived)}

# %% Pool both ASLs using retained-trial frequencies, retaining condition audits.
pooled_hist=[];pooled_curves=[];pooled_psych=[];chronometric=[];weights_rows=[]
condition_curves=tables['condition_model_curves.csv']
for group in groups:
    info=loaded[group];sub=info['sub'];ct=info['conditions'];cond=info['fit']
    for delta in absolute:
        obs=sub.loc[np.isclose(sub.signed_delta.abs(),delta)]
        components=ct.loc[np.isclose(ct.signed_delta.abs(),delta)].sort_values('condition_index')
        weights=components.n.to_numpy()/len(obs)
        assert abs(weights.sum()-1)<1e-14 and len(components)==2
        arrays=[condition_curves.loc[condition_curves.Group.eq(group)&condition_curves.condition_index.eq(i)]
                .sort_values('time_s') for i in components.condition_index]
        density=np.column_stack([c.density for c in arrays])@weights
        cdf=np.column_stack([c.cdf for c in arrays])@weights
        mass=np.trapz(density,times);mean=np.trapz(times*density,times)
        mean_cdf=np.trapz(1-cdf,times)
        assert abs(mass-1)<3e-5 and abs(mean-mean_cdf)<3e-5
        counts=np.histogram(obs.RT_s,edges)[0]
        pooled_hist.append(pd.DataFrame(dict(Group=group,abs_delta=delta,left_s=edges[:-1],right_s=edges[1:],
            count=counts,n=len(obs),density=counts/(len(obs)*.02))))
        pooled_curves.append(pd.DataFrame(dict(Group=group,abs_delta=delta,time_s=times,density=density,cdf=cdf)))
        chronometric.append(dict(Group=group,abs_delta=delta,n=len(obs),data_mean_RT_s=obs.RT_s.mean(),
            model_mean_RT_s=mean,model_mean_from_cdf_s=mean_cdf,model_mass=mass))
        for c,weight in zip(components.itertuples(),weights):
            weights_rows.append(dict(Group=group,panel='rtd',pooled_delta=delta,condition_index=c.condition_index,
                signed_delta=c.signed_delta,component_n=c.n,pooled_n=len(obs),weight=weight))
    for delta in sorted(ct.signed_delta.unique()):
        obs=sub.loc[sub.signed_delta.eq(delta)]
        components=cond.loc[cond.signed_delta.eq(delta)].sort_values('condition_index')
        weights=components.n.to_numpy()/len(obs)
        assert abs(weights.sum()-1)<1e-14 and len(components)==1
        pooled_psych.append(dict(Group=group,signed_delta=delta,n=len(obs),right=int(obs.Choice.eq(1).sum()),
            empirical_p_right=obs.Choice.eq(1).mean(),model_p_right=components.model_p_right.to_numpy()@weights))
        for c,weight in zip(components.itertuples(),weights):
            weights_rows.append(dict(Group=group,panel='psychometric',pooled_delta=delta,condition_index=c.condition_index,
                signed_delta=c.signed_delta,component_n=c.n,pooled_n=len(obs),weight=weight))
tables.update(pooled_rtd_histograms=pd.concat(pooled_hist),pooled_model_curves=pd.concat(pooled_curves),
              pooled_psychometric=pd.DataFrame(pooled_psych),chronometric_points=pd.DataFrame(chronometric),
              mixture_weights=pd.DataFrame(weights_rows))
# Consistent CSV names for saved reconstruction tables.
tables={name if name.endswith('.csv') else name+'.csv':frame for name,frame in tables.items()}
baseline=pd.read_csv(REFERENCE_ROOT/'fit_quality.csv')
baseline['model']='Gamma/delays per signed difference; shared Omega (34 parameters)'
old_chrono=pd.read_csv(REFERENCE_ROOT/'chronometric_points.csv')
old_psych=pd.read_csv(REFERENCE_ROOT/'pooled_psychometric.csv')
old_curves=pd.read_csv(REFERENCE_ROOT/'condition_model_curves.csv')
old_conditions=pd.read_csv(REFERENCE_ROOT/'condition_fit_quality.csv')
for group in groups:
    old_p=old_psych.loc[old_psych.Group.eq(group)]
    error=old_p.model_p_right-old_p.empirical_p_right
    baseline.loc[baseline.Group.eq(group),'psychometric_mae']=abs(error).mean()
    baseline.loc[baseline.Group.eq(group),'psychometric_rmse']=np.sqrt(np.mean(error**2))
    gaps=[]
    for delta in sorted(old_p.signed_delta):
        obs=np.sort(selected.loc[selected.Group.eq(group)&np.isclose(selected.signed_delta,delta),'RT_s'])
        cond=old_conditions.loc[old_conditions.Group.eq(group)&np.isclose(old_conditions.signed_delta,delta)]
        weights=cond.n.to_numpy()/len(obs)
        cdf=np.column_stack([old_curves.loc[old_curves.Group.eq(group)&old_curves.condition_index.eq(i)].sort_values('time_s').cdf for i in cond.condition_index])@weights
        c=np.interp(obs,times,cdf);emp=np.arange(1,len(obs)+1)/len(obs)
        gaps.append(max(abs(c-emp).max(),abs(c-emp+1/len(obs)).max()))
    baseline.loc[baseline.Group.eq(group),'mean_condition_cdf_gap']=np.mean(gaps)
    baseline.loc[baseline.Group.eq(group),'max_condition_cdf_gap']=max(gaps)
for frame,points in [(baseline,old_chrono),(tables['fit_quality.csv'],tables['chronometric_points.csv'])]:
    for group in groups:
        p=points.loc[points.Group.eq(group)]
        frame.loc[frame.Group.eq(group),'pooled_chronometric_rmse_s']=np.sqrt(np.mean((p.model_mean_RT_s-p.data_mean_RT_s)**2))
tables['baseline_vs_shared_gaussian_delay_fit_quality.csv']=pd.concat([baseline,tables['fit_quality.csv']],ignore_index=True)
for name,frame in tables.items():frame.to_csv(out/name,index=False,float_format='%.17g')

# %% Match the seven-panel pooled layout: five RTDs, choice and mean RT.
hist=tables['pooled_rtd_histograms.csv'];curves=tables['pooled_model_curves.csv']
psych=tables['pooled_psychometric.csv'];points=tables['chronometric_points.csv']
ymax=np.ceil(max(hist.density.max(),curves.density.max())*1.08*2)/2
fit_status=' | '.join(f'{g}: best {loaded[g]["report"]["restored_best_step"]:,}; '+
    ('loss confirmation incomplete' if loaded[g]['report']['flagged'] else 'loss confirmation passed') for g in groups)
fig=plt.figure(figsize=(26,6))
grid=fig.add_gridspec(1,7,width_ratios=[1,1,1,1,1,1.55,1.35],left=.035,right=.99,bottom=.28,top=.70,wspace=.45)
for col,delta in enumerate(absolute):
    ax=fig.add_subplot(grid[0,col]);counts=[]
    for group in groups:
        h=hist.loc[hist.Group.eq(group)&hist.abs_delta.eq(delta)]
        m=curves.loc[curves.Group.eq(group)&curves.abs_delta.eq(delta)]
        ax.stairs(h.density.to_numpy(),np.r_[h.left_s,h.right_s.iloc[-1]],color=group_colors[group],alpha=.4,lw=1.25)
        ax.plot(m.time_s,m.density,color=group_colors[group],lw=2.2)
        counts.append(int(h.n.iloc[0]))
    ax.set(xlim=(0,1.8),ylim=(0,ymax),xticks=[0,.6,1.2,1.8],xlabel='RT (s)',
           title=f'|ΔdB| = {delta:g}\nn: Control {counts[0]:,} · ASD {counts[1]:,}')
    ax.grid(alpha=.16)
    if col==0:ax.set_ylabel('Density (s⁻¹)')
    else:ax.tick_params(labelleft=False)
ax=fig.add_subplot(grid[0,5])
for group in groups:
    p=psych.loc[psych.Group.eq(group)].sort_values('signed_delta')
    ax.plot(p.signed_delta,p.model_p_right,color=group_colors[group],lw=2.2,label=group)
    ax.scatter(p.signed_delta,p.empirical_p_right,color=group_colors[group],s=28,edgecolor='white',linewidth=.4,zorder=3)
ax.set(title='Psychometric\nBoth ASLs pooled',xlabel='Signed ΔdB (right − left)',ylabel='P(right)',
       xlim=(-2.15,2.15),ylim=(-.015,1.015),xticks=[-2,-1,0,1,2])
ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9)
ax=fig.add_subplot(grid[0,6])
for group in groups:
    p=points.loc[points.Group.eq(group)].sort_values('abs_delta')
    ax.plot(p.abs_delta,p.model_mean_RT_s,color=group_colors[group],lw=2.2)
    ax.scatter(p.abs_delta,p.data_mean_RT_s,color=group_colors[group],s=28,edgecolor='white',linewidth=.4,zorder=3)
ax.set(title='Chronometric\nBoth ASLs pooled',xlabel='|ΔdB|',ylabel='Mean RT (s)',
       xticks=absolute,xticklabels=['0.1','0.3','0.65','1.3','2'],xlim=(0,2.1))
v=points[['data_mean_RT_s','model_mean_RT_s']].to_numpy()
ax.set_ylim(np.floor((v.min()-.035)*20)/20,np.ceil((v.max()+.035)*20)/20);ax.grid(alpha=.18)
fig.suptitle('Human simultaneous RT | one shared Gaussian delay per group | ASLs pooled in fitting',fontsize=19,y=.97)
fig.text(.5,.90,'Control: 6,574 trials, 11 participants   |   ASD: 4,770 trials, 8 participants   |   ASL 20 + 40 cd/m²   |   16 parameters per group',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='.45',lw=1.4,alpha=.6,label='Data: 20 ms RT bins'),
    Line2D([],[],color='.45',marker='o',linestyle='none',label='Data: choice proportions / mean RT'),
    Line2D([],[],color='.2',lw=2.2,label='Model at posterior means'),
    *[Line2D([],[],color=group_colors[g],lw=2.2,label=g) for g in groups]],
    loc='upper center',bbox_to_anchor=(.5,.855),ncol=5,frameon=False)
fig.text(.5,.15,'Condition = RT; Valid = 1; 0–1.8 s retained | Both choices; raw trials pooled across participants, sessions and ASLs',ha='center',fontsize=10)
fig.text(.5,.105,'Gamma per signed ΔdB | One Omega and Gaussian delay μ/σ across all stimuli within each group | Uniform lapses on 0–1.8 s',ha='center',fontsize=10)
fig.text(.5,.06,'RTDs and chronometrics pool ±Δ with retained-trial weights | Model points joined only at measured differences',ha='center',fontsize=10)
fig.text(.5,.015,fit_status,ha='center',fontsize=10)
figures=[]
for suffix in ['png','pdf']:
    name=f'human_shared_gaussian_delay_pooled_comparison.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)

# %% Gamma by signed difference, plus six genuinely shared scalars.
fig=plt.figure(figsize=(14,12))
outer=fig.add_gridspec(3,3,height_ratios=[1.65,1,1],left=.08,right=.97,bottom=.11,top=.85,hspace=.65,wspace=.42)
ax=fig.add_subplot(outer[0,:])
subset=tables['parameter_summary.csv'].loc[lambda x:x.block.eq('gamma')]
for group,shift in [('Control',-.015),('ASD',.015)]:
    p=subset.loc[subset.Group.eq(group)].sort_values('signed_delta');y=p['mean'].to_numpy()
    ax.errorbar(p.signed_delta+shift,y,yerr=np.vstack([y-p.q025.to_numpy(),p.q975.to_numpy()-y]),
        fmt='o-',color=group_colors[group],ms=4,lw=1.4,capsize=3,label=group)
ax.set(ylabel='Gamma',xlabel='Signed ΔdB (right − left)',xticks=[-2,-1,0,1,2],xlim=(-2.15,2.15))
ax.axhline(0,color='.6',lw=.7);ax.legend(frameon=False);ax.grid(alpha=.18)
scalars=[('omega','Shared Omega (s⁻¹)',1),('delay_mu','Shared Gaussian μ (ms)',1000),
    ('delay_sigma','Shared Gaussian σ (ms)',1000),('w','Starting bias w',1),
    ('lapse_prob','Lapse probability',1),('lapse_prob_right','Lapse-right probability',1)]
for index,(name,label,factor) in enumerate(scalars):
    ax=fig.add_subplot(outer[1+index//3,index%3]);bounds=[]
    for pos,group in enumerate(groups):
        p=loaded[group]['summary'].loc[lambda x:x.parameter.eq(name)].iloc[0]
        y,lo,hi=p['mean']*factor,p.q025*factor,p.q975*factor
        ax.errorbar(pos,y,yerr=[[y-lo],[hi-y]],fmt='o',color=group_colors[group],capsize=4)
        bounds.extend([lo,hi])
    pad=max((max(bounds)-min(bounds))*.25,.002)
    ax.set(title=label,xlim=(-.5,1.5),xticks=[0,1],xticklabels=groups,ylim=(min(bounds)-pad,max(bounds)+pad));ax.grid(axis='y',alpha=.18)
fig.suptitle('Human RT | one shared Gaussian delay per group | parameter comparison',fontsize=18,y=.975)
fig.text(.5,.935,'Control blue · ASD red | Posterior means and equal-tailed 95% variational credible intervals',ha='center',fontsize=11)
fig.text(.5,.895,'16 parameters per group | Gamma per signed ΔdB; Omega, Gaussian μ/σ, bias and lapses shared across stimuli',ha='center',fontsize=10)
fig.text(.5,.065,'μ and σ describe trial-varying delays: Gaussian truncated below zero, with no upper cutoff.',ha='center',fontsize=10)
fig.text(.5,.04,'Pooled-trial variational approximation; intervals do not represent participant-to-participant variability.',ha='center',fontsize=10)
fig.text(.5,.015,fit_status,ha='center',fontsize=10)
for suffix in ['png','pdf']:
    name=f'human_shared_gaussian_delay_parameter_comparison.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)

# %% Loss/correlation diagnostics and reproducible output inventory.
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for row,group in enumerate(groups):
    trace=pd.read_csv(out/group.lower()/'convergence.csv');report=loaded[group]['report']
    axes[row,0].plot(trace.step,trace.mean_negative_elbo,color=group_colors[group])
    axes[row,0].axvline(report['restored_best_step'],color='k',ls='--',label='Restored minimum')
    axes[row,0].set(title=f'{group} | negative ELBO',xlabel='Updates',ylabel='1,000-update average');axes[row,0].legend()
    im=axes[row,1].imshow(loaded[group]['corr'],vmin=-1,vmax=1,cmap='coolwarm')
    axes[row,1].set(title=f'{group} | 16-parameter posterior correlation',xlabel='Parameter index',ylabel='Parameter index')
fig.colorbar(im,ax=axes[:,1],shrink=.75)
fig.savefig(out/'posterior_diagnostics.png',dpi=150);plt.close(fig)
manifest=dict(source=str(Path(__file__).relative_to(ROOT)),source_sha256=sha256(__file__),
    reference_fit=str(REFERENCE_ROOT.relative_to(ROOT)),selection='Condition == RT, Valid == 1, 0 <= RT_s <= 1.8',
    trials=11344,parameters_per_group=16,curve_parameters='arithmetic physical posterior means',
    parameter_order=PARAMETER_NAMES,omega='one scalar shared across both ASLs and all signed differences per group',
    pooling='raw retained-trial frequency weights',delay='one scalar mu and sigma per group; identical distribution across ten differences',figures=figures,
    source_hashes=config['source_hashes'],files={n:sha256(out/n) for n in list(tables)+figures+['posterior_diagnostics.png']})
(out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(tables['baseline_vs_shared_gaussian_delay_fit_quality.csv'].to_string(index=False),flush=True)
