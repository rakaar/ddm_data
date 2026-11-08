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
from human_gaussian_delay_svi_utils import *

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
    assert matrix.shape==(10000,83)
    params={k:jnp.asarray(v) for k,v in vector_params(matrix.mean(0)).items()}
    sub,ct,data,_,_=prepare_group(selected,group)
    summary=pd.read_csv(folder/'posterior_summary.csv')
    np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=1e-12)
    summary.insert(0,'Group',group)
    summary['block']=[n for n in PARAMETER_BLOCKS[:4] for _ in range(20)]+list(PARAMETER_BLOCKS[4:])
    summary['condition_index']=list(range(20))*4+[np.nan]*3
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
    quality.append(dict(Group=group,model='Gaussian delay + choice + uniform lapse',free_parameters=83,
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
        boundaries.append(dict(Group=group,parameter=name,hard_low=lo,hard_high=hi,
            fraction_within_1pct_lower=float(np.mean(matrix[:,i]<lo+.01*(hi-lo))),
            fraction_within_1pct_upper=float(np.mean(matrix[:,i]>hi-.01*(hi-lo)))))
    corr=np.corrcoef(matrix,rowvar=False);ia,ib=np.triu_indices(83,1)
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
baseline=pd.read_csv(BASELINE_ROOT/'fit_quality.csv')
baseline['model']='Fixed delay + choice + uniform lapse';baseline['free_parameters']=63
tables['fixed_vs_gaussian_fit_quality.csv']=pd.concat([baseline,tables['fit_quality.csv']],ignore_index=True)
for name,df in tables.items():df.to_csv(out/name,index=False,float_format='%.17g')
hist,curves=tables['rtd_histograms.csv'],tables['rtd_model_curves.csv']
ymax=np.ceil(max(hist.density.max(),curves.density.max())*1.08*2)/2
figures=[]

# %% Identical Control and ASD two-ASL by five-absolute-difference layouts.
for group in groups:
    info=loaded[group];fig=plt.figure(figsize=(21.8,7.6))
    grid=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1,1.7],left=.055,right=.985,bottom=.18,top=.79,hspace=.38,wspace=.33)
    for r,level in enumerate(levels):
        for c,delta in enumerate(absolute):
            ax=fig.add_subplot(grid[r,c])
            h=hist.loc[hist.Group.eq(group)&hist.ASL_cd.eq(level)&hist.abs_delta.eq(delta)]
            m=curves.loc[curves.Group.eq(group)&curves.ASL_cd.eq(level)&curves.abs_delta.eq(delta)]
            ax.stairs(h.density.to_numpy(),np.r_[h.left_s,h.right_s.iloc[-1]],color=level_colors[level],alpha=.55,lw=1.5)
            ax.plot(m.time_s,m.density,color=level_colors[level],lw=2.1)
            ax.set(xlim=(0,1.8),ylim=(0,ymax),xticks=[0,.6,1.2,1.8],title=f'|ΔdB| = {delta:g} · n={int(h.n.iloc[0])}')
            ax.grid(alpha=.16)
            if c==0:ax.set_ylabel(f'ASL {level:g} cd/m²\nDensity (s⁻¹)')
            else:ax.tick_params(labelleft=False)
            if r==1:ax.set_xlabel('RT (s)')
    ax=fig.add_subplot(grid[:,5])
    for level in levels:
        p=info['fit'].loc[lambda x:x.ASL_cd.eq(level)].sort_values('signed_delta')
        ax.scatter(p.signed_delta,p.empirical_p_right,color=level_colors[level],s=25,zorder=3)
        ax.plot(p.signed_delta,p.model_p_right,color=level_colors[level],lw=2,label=f'ASL {level:g} cd/m²')
    ax.set(title='Psychometric',xlabel='Signed ΔdB (right − left)',ylabel='P(right)',xlim=(-2.15,2.15),ylim=(-.015,1.015),xticks=[-2,-1,0,1,2])
    ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle(f'{group} | human simultaneous RT | {len(info["sub"]):,} trials from {info["sub"].Participant.nunique()} participants pooled',color=group_colors[group],fontsize=18,fontweight='bold',y=.97)
    fig.text(.5,.915,'RT + choice + uniform lapses | 20 Gamma–Omega–Gaussian μ–σ sets + shared bias and lapse probabilities (83 parameters)',ha='center',fontsize=10.5)
    fig.legend(handles=[Line2D([],[],color='.5',lw=1.5,alpha=.65,drawstyle='steps-mid',label='Data: 20 ms bins / choice dots'),
                        Line2D([],[],color='.2',lw=2.1,label='Model at arithmetic posterior means')],loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False)
    status='Convergence checks passed' if not info['report']['flagged'] else 'PROVISIONAL: convergence checks not satisfied'
    fig.text(.5,.103,'Condition = RT; 0–1.8 s retained, both choices; raw trials pooled across participants and sessions',ha='center',fontsize=10)
    fig.text(.5,.065,'Independent Gaussian delay per trial, truncated at zero with no upper cutoff | lapse RTs uniform on 0–1.8 s',ha='center',fontsize=10)
    fig.text(.5,.027,f'{status} | ±Δ pooled using retained counts; model psychometric points joined only as visual guides',ha='center',fontsize=10)
    stem=f'human_{group.lower()}_gaussian_delay_rtd_psychometric'
    for suffix in ['png','pdf']:
        fig.savefig(out/f'{stem}.{suffix}',dpi=180);figures.append(f'{stem}.{suffix}')
    plt.close(fig)

# %% Four condition-parameter rows and three shared scalar panels.
fig=plt.figure(figsize=(18,16))
grid=fig.add_gridspec(4,3,width_ratios=[1,1,.65],left=.07,right=.975,bottom=.10,top=.91,wspace=.32,hspace=.45)
for r,(block,label,scale) in enumerate([('gamma','Gamma',1.),('omega','Omega (s⁻¹)',1.),
                                      ('delay_mu','Gaussian μ (ms)',1000.),('delay_sigma','Gaussian σ (ms)',1000.)]):
    axes=[]
    for c,level in enumerate(levels):
        ax=fig.add_subplot(grid[r,c],sharey=axes[0] if axes else None);axes.append(ax)
        for group,offset,marker in [('Control',-.018,'o'),('ASD',.018,'*')]:
            p=loaded[group]['summary'].loc[lambda x:x.block.eq(block)&x.ASL_cd.eq(level)].sort_values('signed_delta')
            mean=p['mean'].to_numpy()*scale
            ax.errorbar(p.signed_delta+offset,mean,yerr=np.vstack([mean-p.q025.to_numpy()*scale,p.q975.to_numpy()*scale-mean]),fmt=marker,color=group_colors[group],ms=4.5 if group=='Control' else 7,capsize=2,lw=1.1,label=group)
        ax.set(xticks=sorted(selected.signed_delta.unique()),xlim=(-2.14,2.14),ylabel=label if c==0 else '')
        ax.tick_params(axis='x',rotation=55,labelsize=8);ax.grid(alpha=.18)
        if r==0:ax.set_title(f'ASL {level:g} cd/m²');ax.axhline(0,color='.5',lw=.8)
        if r==3:ax.set_xlabel('Signed ΔdB (right − left)')
    interval=tables['parameter_summary.csv'].loc[lambda x:x.block.eq(block)]
    low=float(interval.q025.min())*scale;high=float(interval.q975.max())*scale
    margin=max(.08*(high-min(0,low)),.01)
    axes[0].set_ylim(min(0,low-margin) if block=='gamma' else 0,high+margin)
    if r==0:axes[0].legend(frameon=False)
for r,(name,title,high) in enumerate([('w','Shared bias w',1.),('lapse_prob','Shared lapse probability p',.4),('lapse_prob_right','P(right | lapse)',1.)]):
    ax=fig.add_subplot(grid[r,2])
    for i,group in enumerate(groups):
        p=loaded[group]['summary'].loc[lambda x:x.parameter.eq(name)].iloc[0]
        ax.errorbar(i,p['mean'],yerr=[[p['mean']-p.q025],[p.q975-p['mean']]],fmt='o' if group=='Control' else '*',color=group_colors[group],capsize=4,ms=7,lw=1.8)
        ax.text(i,p.q975+.025*high,f'{p["mean"]:.3f}\n[{p.q025:.3f}, {p.q975:.3f}]',ha='center',va='bottom',fontsize=8,color=group_colors[group])
    ax.set(title=title,xticks=[0,1],xticklabels=groups,xlim=(-.6,1.6),ylim=(0,high*1.13));ax.grid(axis='y',alpha=.18)
ax=fig.add_subplot(grid[3,2]);ax.axis('off')
ax.text(0,.9,'μ and σ describe the\nunderlying Gaussian.\n\nDelay draws are truncated\nat zero, without an upper cap.\n\nActual truncated mean/SD\nare provided in the tables.',va='top',fontsize=11,linespacing=1.6)
fig.suptitle('Human Gaussian-delay + choice + lapse | posterior comparison',fontsize=20,y=.98)
fig.text(.5,.945,'Control blue; ASD red | posterior means and equal-tailed 95% variational credible intervals | 83 parameters per group',ha='center',fontsize=11)
fig.text(.5,.045,'Pooled-trial variational uncertainty does not represent participant-to-participant variability.',ha='center',fontsize=11)
for suffix in ['png','pdf']:
    name=f'human_gaussian_delay_parameter_comparison.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)

# %% Supplementary convergence and correlation evidence.
fig,axes=plt.subplots(2,2,figsize=(14,10))
for r,group in enumerate(groups):
    trace=pd.read_csv(out/group.lower()/'convergence.csv');report=loaded[group]['report']
    axes[r,0].plot(trace.step,trace.mean_negative_elbo,lw=1)
    axes[r,0].axvline(report['restored_best_step'],color='red',ls='--',label='Restored best')
    axes[r,0].set(title=f'{group} | final {report["completed_steps"]:,}; restored {report["restored_best_step"]:,}',xlabel='SVI updates',ylabel='1000-step mean negative ELBO')
    axes[r,0].legend();im=axes[r,1].imshow(loaded[group]['corr'],vmin=-1,vmax=1,cmap='coolwarm')
    axes[r,1].set(title=f'{group} | 83-parameter posterior correlation',xlabel='Parameter index',ylabel='Parameter index')
fig.subplots_adjust(left=.08,right=.87,hspace=.3,wspace=.3)
colorbar_axis=fig.add_axes([.91,.17,.015,.66])
fig.colorbar(im,cax=colorbar_axis,label='Pearson correlation')
for suffix in ['png','pdf']:
    name=f'human_gaussian_delay_convergence_correlations.{suffix}';fig.savefig(out/name,dpi=170);figures.append(name)
plt.close(fig)
manifest=dict(figures=figures,tables=list(tables),source_sha256=sha256(Path(__file__)),
    source=str(Path(__file__).relative_to(ROOT)),condition='RT',retention_window_s=[0,1.8],
    trials=11344,parameters_per_group=83,curve_parameters='arithmetic physical posterior means',
    psychometric='empirical dots without error bars; model points joined at measured differences',
    baseline_root=str(BASELINE_ROOT.relative_to(ROOT)),baseline_quality_sha256=sha256(BASELINE_ROOT/'fit_quality.csv'))
(out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(tables['fixed_vs_gaussian_fit_quality.csv'].to_string(index=False),flush=True)
print('Saved',len(figures),'figure files',flush=True)
