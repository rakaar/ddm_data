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
from human_shared_omega_gaussian_delay_svi_utils import *

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
    assert matrix.shape==(10000,64)
    params={k:jnp.asarray(v) for k,v in vector_params(matrix.mean(0)).items()}
    sub,ct,data,_,_=prepare_group(selected,group)
    summary=pd.read_csv(folder/'posterior_summary.csv')
    np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=1e-12)
    summary.insert(0,'Group',group)
    summary['block']=[n for n in PARAMETER_BLOCKS for _ in (range(20) if n in CONDITION_BLOCKS else [0])]
    summary['condition_index']=[i for n in PARAMETER_BLOCKS for i in (range(20) if n in CONDITION_BLOCKS else [np.nan])]
    summary=summary.merge(ct[['condition_index','ASL_cd','signed_delta']],on='condition_index',how='left',validate='many_to_one')
    summaries.append(summary)
    prediction=prediction_components(params,times)
    rows=[]
    for c in ct.itertuples():
        i=c.condition_index;rt=np.sort(sub.loc[sub.condition_index.eq(i),'RT_s'].to_numpy())
        model_cdf=np.interp(rt,times,prediction['cdf'][:,i])
        empirical=np.arange(1,len(rt)+1)/len(rt)
        gap=float(max(abs(model_cdf-empirical).max(),abs(model_cdf-empirical+1/len(rt)).max()))
        rows.append(dict(Group=group,condition_index=i,ASL_cd=c.ASL_cd,signed_delta=c.signed_delta,
            n=c.n,right=c.right,empirical_p_right=c.right/c.n,model_p_right=prediction['p_right'][i],
            cdf_gap=gap,retained_mass=prediction['retained_mass'][i]))
        curve_rows.append(pd.DataFrame(dict(Group=group,condition_index=i,ASL_cd=c.ASL_cd,
            signed_delta=c.signed_delta,time_s=times,**{k:prediction[k][:,i]
            for k in ['left_density','right_density','density','cdf']})))
    cond=pd.DataFrame(rows);condition_rows.append(cond)
    joint=float(jax.jit(lambda p:log_likelihood(p,data))(params))
    marginal=float(jax.jit(lambda p:log_likelihood(p,data,marginal=True))(params))
    errors=cond.model_p_right-cond.empirical_p_right
    quality.append(dict(Group=group,model='Shared Omega + Gaussian delay + choice + uniform lapse',free_parameters=64,
        n_trials=len(sub),joint_log_likelihood=joint,marginal_rt_log_likelihood=marginal,
        psychometric_mae=float(abs(errors).mean()),psychometric_rmse=float(np.sqrt(np.mean(errors**2))),
        mean_condition_cdf_gap=float(cond.cdf_gap.mean()),max_condition_cdf_gap=float(cond.cdf_gap.max())))
    for level in levels:
        for delta in absolute:
            obs=sub.loc[sub.ASL_cd.eq(level)&np.isclose(sub.signed_delta.abs(),delta),'RT_s'].to_numpy()
            pair=ct.loc[ct.ASL_cd.eq(level)&np.isclose(ct.signed_delta.abs(),delta)].sort_values('signed_delta')
            ids=pair.condition_index.to_numpy(int);weights=pair.n.to_numpy()/len(obs)
            counts=np.histogram(obs,edges)[0]
            hists.append(pd.DataFrame(dict(Group=group,ASL_cd=level,abs_delta=delta,left_s=edges[:-1],
                right_s=edges[1:],count=counts,n=len(obs),density=counts/(len(obs)*.02))))
            mixtures.append(pd.DataFrame(dict(Group=group,ASL_cd=level,abs_delta=delta,time_s=times,
                density=prediction['density'][:,ids]@weights,cdf=prediction['cdf'][:,ids]@weights,
                negative_weight=weights[0],positive_weight=weights[1],negative_n=pair.n.iloc[0],positive_n=pair.n.iloc[1])))
    for i,name in enumerate(PARAMETER_NAMES):
        key=summary.block.iloc[i]
        if key=='gamma':key='gamma_positive' if ct.signed_delta.iloc[i]>0 else 'gamma_negative'
        lo,hi=PRIORS[key]['hard']
        boundaries.append(dict(Group=group,parameter=name,hard_low=lo,hard_high=hi,q005=float(np.quantile(matrix[:,i],.005)),q995=float(np.quantile(matrix[:,i],.995)),
            lower_tail_distance_fraction=float((np.quantile(matrix[:,i],.005)-lo)/(hi-lo)),
            upper_tail_distance_fraction=float((hi-np.quantile(matrix[:,i],.995))/(hi-lo)),
            fraction_within_1pct_lower=float(np.mean(matrix[:,i]<lo+.01*(hi-lo))),
            fraction_within_1pct_upper=float(np.mean(matrix[:,i]>hi-.01*(hi-lo)))))
    corr=np.corrcoef(matrix,rowvar=False);ia,ib=np.triu_indices(64,1)
    for j in np.argsort(-abs(corr[ia,ib]))[:20]:
        correlations.append(dict(Group=group,parameter_1=PARAMETER_NAMES[ia[j]],parameter_2=PARAMETER_NAMES[ib[j]],correlation=corr[ia[j],ib[j]]))
    a=posterior['delay_mu']/posterior['delay_sigma']
    ratio=np.exp(-.5*a*a)/(np.sqrt(2*np.pi)*ndtr(a))
    actual_mean=posterior['delay_mu']+posterior['delay_sigma']*ratio
    actual_sd=posterior['delay_sigma']*np.sqrt(1-a*ratio-ratio*ratio)
    for label,values in [('actual_delay_mean',actual_mean),('actual_delay_sd',actual_sd)]:
        for c in ct.itertuples():
            x=values[:,c.condition_index]
            derived.append(dict(Group=group,quantity=label,condition_index=c.condition_index,ASL_cd=c.ASL_cd,
                signed_delta=c.signed_delta,mean_s=x.mean(),posterior_sd_s=x.std(ddof=1),q025_s=np.quantile(x,.025),q975_s=np.quantile(x,.975)))
    loaded[group]=dict(summary=summary,conditions=ct,sub=sub,fit=cond,posterior=posterior,corr=corr,
        report=json.loads((folder/'convergence_summary.json').read_text()))

tables={'parameter_summary.csv':pd.concat(summaries,ignore_index=True),
        'fit_quality.csv':pd.DataFrame(quality),'condition_fit_quality.csv':pd.concat(condition_rows,ignore_index=True),
        'psychometric_comparison.csv':pd.concat(condition_rows,ignore_index=True),
        'condition_model_curves.csv':pd.concat(curve_rows,ignore_index=True),
        'rtd_histograms.csv':pd.concat(hists,ignore_index=True),'rtd_model_curves.csv':pd.concat(mixtures,ignore_index=True),
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
        assert abs(weights.sum()-1)<1e-14 and len(components)==4
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
                ASL_cd=c.ASL_cd,signed_delta=c.signed_delta,component_n=c.n,pooled_n=len(obs),weight=weight))
    for delta in sorted(ct.signed_delta.unique()):
        obs=sub.loc[sub.signed_delta.eq(delta)]
        components=cond.loc[cond.signed_delta.eq(delta)].sort_values('condition_index')
        weights=components.n.to_numpy()/len(obs)
        assert abs(weights.sum()-1)<1e-14 and len(components)==2
        pooled_psych.append(dict(Group=group,signed_delta=delta,n=len(obs),right=int(obs.Choice.eq(1).sum()),
            empirical_p_right=obs.Choice.eq(1).mean(),model_p_right=components.model_p_right.to_numpy()@weights))
        for c,weight in zip(components.itertuples(),weights):
            weights_rows.append(dict(Group=group,panel='psychometric',pooled_delta=delta,condition_index=c.condition_index,
                ASL_cd=c.ASL_cd,signed_delta=c.signed_delta,component_n=c.n,pooled_n=len(obs),weight=weight))
tables.update(pooled_rtd_histograms=pd.concat(pooled_hist),pooled_model_curves=pd.concat(pooled_curves),
              pooled_psychometric=pd.DataFrame(pooled_psych),chronometric_points=pd.DataFrame(chronometric),
              mixture_weights=pd.DataFrame(weights_rows))
# Consistent CSV names for saved reconstruction tables.
tables={name if name.endswith('.csv') else name+'.csv':frame for name,frame in tables.items()}
baseline=pd.read_csv(REFERENCE_ROOT/'fit_quality.csv')
baseline['model']='Condition-wise Omega (83 parameters)';baseline['free_parameters']=83
old_chrono=pd.read_csv(REFERENCE_ROOT/'pooled_asl_chronometric/chronometric_points.csv')
for frame,points in [(baseline,old_chrono),(tables['fit_quality.csv'],tables['chronometric_points.csv'])]:
    for group in groups:
        p=points.loc[points.Group.eq(group)]
        frame.loc[frame.Group.eq(group),'pooled_chronometric_rmse_s']=np.sqrt(np.mean((p.model_mean_RT_s-p.data_mean_RT_s)**2))
tables['baseline_vs_shared_omega_fit_quality.csv']=pd.concat([baseline,tables['fit_quality.csv']],ignore_index=True)
for name,frame in tables.items():frame.to_csv(out/name,index=False,float_format='%.17g')

# %% Match the seven-panel pooled layout: five RTDs, choice and mean RT.
hist=tables['pooled_rtd_histograms.csv'];curves=tables['pooled_model_curves.csv']
psych=tables['pooled_psychometric.csv'];points=tables['chronometric_points.csv']
ymax=np.ceil(max(hist.density.max(),curves.density.max())*1.08*2)/2
fit_status=' | '.join(f'{g}: best {loaded[g]["report"]["restored_best_step"]:,}; '+
    ('parameter stability unconfirmed' if loaded[g]['report']['flagged'] else 'terminal stability passed') for g in groups)
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
fig.suptitle('Human simultaneous RT | one shared Omega per group | both ASLs pooled',fontsize=19,y=.97)
fig.text(.5,.90,'Control: 6,574 trials, 11 participants   |   ASD: 4,770 trials, 8 participants   |   ASL 20 + 40 cd/m²   |   64 parameters per group',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='.45',lw=1.4,alpha=.6,label='Data: 20 ms RT bins'),
    Line2D([],[],color='.45',marker='o',linestyle='none',label='Data: choice proportions / mean RT'),
    Line2D([],[],color='.2',lw=2.2,label='Model at posterior means'),
    *[Line2D([],[],color=group_colors[g],lw=2.2,label=g) for g in groups]],
    loc='upper center',bbox_to_anchor=(.5,.855),ncol=5,frameon=False)
fig.text(.5,.15,'Condition = RT; Valid = 1; 0–1.8 s retained | Both choices; raw trials pooled across participants, sessions and ASLs',ha='center',fontsize=10)
fig.text(.5,.105,'One Omega across all 20 conditions within each group | Condition-wise Gamma and Gaussian delay μ/σ | Uniform lapses on 0–1.8 s',ha='center',fontsize=10)
fig.text(.5,.06,'RTDs and chronometrics pool ±Δ with retained-trial weights | Model points joined only at measured differences',ha='center',fontsize=10)
fig.text(.5,.015,fit_status,ha='center',fontsize=10)
figures=[]
for suffix in ['png','pdf']:
    name=f'human_shared_omega_pooled_comparison.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)

# %% Condition-wise parameters on the left; four genuinely shared scalars on the right.
fig=plt.figure(figsize=(16,12))
outer=fig.add_gridspec(1,2,width_ratios=[4.5,1],left=.07,right=.96,bottom=.10,top=.88,wspace=.33)
grid=outer[0,0].subgridspec(3,2,hspace=.36,wspace=.24)
for row,(block,label,factor) in enumerate([('gamma','Gamma',1),('delay_mu','Gaussian μ (ms)',1000),('delay_sigma','Gaussian σ (ms)',1000)]):
    subset=tables['parameter_summary.csv'].loc[lambda x:x.block.eq(block)]
    bounds=np.r_[subset.q025,subset.q975]*factor
    padding=max((bounds.max()-bounds.min())*.08,.005)
    for col,level in enumerate(levels):
        ax=fig.add_subplot(grid[row,col])
        for group,shift in [('Control',-.018),('ASD',.018)]:
            p=subset.loc[subset.Group.eq(group)&subset.ASL_cd.eq(level)].sort_values('signed_delta')
            y=p['mean'].to_numpy()*factor
            ax.errorbar(p.signed_delta+shift,y,yerr=np.vstack([y-p.q025.to_numpy()*factor,p.q975.to_numpy()*factor-y]),
                        fmt='o-',color=group_colors[group],ms=4,lw=1.3,capsize=2,label=group)
        ax.set(title=f'ASL {level:g} cd/m²' if row==0 else '',ylabel=label,xlim=(-2.15,2.15),
               ylim=(bounds.min()-padding,bounds.max()+padding),xticks=[-2,-1,0,1,2])
        if row==2:ax.set_xlabel('Signed ΔdB (right − left)')
        if row==0:ax.axhline(0,color='.6',lw=.7)
        ax.grid(alpha=.18)
scalar_grid=outer[0,1].subgridspec(4,1,hspace=.65)
for row,(name,label) in enumerate([('omega','Shared Omega (s⁻¹)'),('w','Starting bias w'),
                                    ('lapse_prob','Lapse probability'),('lapse_prob_right','Lapse-right probability')]):
    ax=fig.add_subplot(scalar_grid[row,0]);bounds=[]
    for pos,group in enumerate(groups):
        p=loaded[group]['summary'].loc[lambda x:x.parameter.eq(name)].iloc[0]
        ax.errorbar(pos,p['mean'],yerr=[[p['mean']-p.q025],[p.q975-p['mean']]],fmt='o',color=group_colors[group],capsize=4)
        bounds.extend([p.q025,p.q975])
    pad=max((max(bounds)-min(bounds))*.25,.002)
    ax.set(title=label,xlim=(-.5,1.5),xticks=[0,1],xticklabels=groups,ylim=(min(bounds)-pad,max(bounds)+pad))
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Human RT | shared-Omega fit | parameter comparison',fontsize=19,y=.975)
fig.text(.5,.937,'Control blue · ASD red | Posterior means and equal-tailed 95% variational credible intervals',ha='center',fontsize=12)
fig.text(.5,.908,'64 parameters per group | Omega, starting bias and lapse probabilities shared across stimuli within each group',ha='center',fontsize=11)
fig.text(.5,.057,'μ and σ describe the underlying Gaussian; trial delays are truncated below zero, with no upper cutoff.',ha='center',fontsize=11)
fig.text(.5,.034,'Pooled-trial variational approximation; intervals do not represent participant-to-participant variability.',ha='center',fontsize=10)
fig.text(.5,.011,fit_status,ha='center',fontsize=10)
for suffix in ['png','pdf']:
    name=f'human_shared_omega_parameter_comparison.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)

# %% Loss/correlation diagnostics and reproducible output inventory.
fig,axes=plt.subplots(2,3,figsize=(18,8),constrained_layout=True)
for row,group in enumerate(groups):
    trace=pd.read_csv(out/group.lower()/'convergence.csv');report=loaded[group]['report']
    axes[row,0].plot(trace.step,trace.mean_negative_elbo,color=group_colors[group])
    axes[row,0].axvline(report['restored_best_step'],color='k',ls='--',label='Restored best')
    axes[row,0].set(title=f'{group} | negative ELBO',xlabel='Updates',ylabel='1,000-update mean');axes[row,0].legend()
    finite=trace.loc[np.isfinite(trace.maximum_mean_change_sd)]
    axes[row,1].semilogy(finite.step,finite.maximum_mean_change_sd,color=group_colors[group])
    axes[row,1].axhline(.1,color='k',ls=':',label='Required < 0.1 SD')
    axes[row,1].axvline(report['restored_best_step'],color='k',ls='--',label='Restored best')
    axes[row,1].set(title=f'{group} | posterior-mean stability',xlabel='Updates',ylabel='Largest range / SD across last 3 checkpoints')
    axes[row,1].legend(fontsize=8);axes[row,1].grid(alpha=.15)
    im=axes[row,2].imshow(loaded[group]['corr'],vmin=-1,vmax=1,cmap='coolwarm')
    axes[row,2].set(title=f'{group} | 64-parameter posterior correlation',xlabel='Parameter index',ylabel='Parameter index')
fig.colorbar(im,ax=axes[:,2],shrink=.75)
fig.savefig(out/'posterior_diagnostics.png',dpi=150);plt.close(fig)
manifest=dict(source=str(Path(__file__).relative_to(ROOT)),source_sha256=sha256(__file__),
    reference_fit=str(REFERENCE_ROOT.relative_to(ROOT)),selection='Condition == RT, Valid == 1, 0 <= RT_s <= 1.8',
    trials=11344,parameters_per_group=64,curve_parameters='arithmetic physical posterior means',
    parameter_order=PARAMETER_NAMES,omega='one scalar shared across both ASLs and all signed differences per group',
    pooling='raw retained-trial frequency weights',figures=figures,
    source_hashes=config['source_hashes'],files={n:sha256(out/n) for n in list(tables)+figures+['posterior_diagnostics.png']})
(out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(tables['baseline_vs_shared_omega_fit_quality.csv'].to_string(index=False),flush=True)
