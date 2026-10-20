# %%
"""Analytic mean-parameter RTDs, psychometrics, posteriors and pooled-group comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_gamma_omega_mpl')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp
from human_gamma_omega_delay500_svi_utils import (
    ROOT, OUTPUT, INITIAL_ROOT, PRIORS, PARAMETER_BLOCKS, PARAMETER_NAMES, RT_WINDOW,
    sha256, load_data, prepare_group, sample_matrix, vector_params, log_likelihood,
    ddm_log_density, ddm_log_cdf, CDF_E_gamma_omega_with_w_jax,
)
from human_plain_ddm_diagnostics_utils import tied_coordinates

# %% Repeated assessment: same trials, joint likelihood and window for every model.
def assess(params, data, sub, conditions, label, group):
    p = {k:jnp.asarray(v) for k,v in params.items()}
    g, o, tau, w = (p[n] for n in PARAMETER_BLOCKS)
    mass = jnp.exp(ddm_log_cdf(1.8,g,o,w,tau))
    right = np.asarray(CDF_E_gamma_omega_with_w_jax(1.8-tau,g,o,1,w,40)/mass)
    ids = np.asarray(data['condition'])
    cdf = np.asarray(jnp.exp(ddm_log_cdf(data['rt'],g[ids],o[ids],w,tau[ids])-jnp.log(mass)[ids]))
    rows = []
    for c in conditions.itertuples():
        i = c.condition_index
        order = np.argsort(sub.loc[ids==i,'RT_s'].to_numpy())
        fitted = cdf[ids==i][order]
        empirical = np.arange(1,c.n+1)/c.n
        gap = float(max(np.abs(fitted-empirical).max(), np.abs(fitted-(empirical-1/c.n)).max()))
        rows.append(dict(Group=group, model=label, condition_index=i, ASL_cd=c.ASL_cd,
                         signed_delta=c.signed_delta, n=c.n, right=c.right, empirical_p_right=c.right/c.n,
                         model_p_right=right[i], retained_mass=float(mass[i]), cdf_gap=gap))
    table = pd.DataFrame(rows)
    errors = table.model_p_right-table.empirical_p_right
    score = dict(Group=group, model=label, n_trials=len(sub),
                 joint_log_likelihood=float(log_likelihood(p,data)),
                 psychometric_mae=float(np.abs(errors).mean()), psychometric_rmse=float(np.sqrt((errors**2).mean())),
                 mean_condition_cdf_gap=float(table.cdf_gap.mean()), max_condition_cdf_gap=float(table.cdf_gap.max()))
    return table, score


# %% Configuration, exact posterior draws and analytic predictions.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
parser.add_argument('--group-figures-only',action='store_true',help='Update the Control/ASD pages while preserving parameter and convergence figures')
args=parser.parse_args()
out=args.output_dir
config=json.loads((out/'run_config.json').read_text())
for path,digest in config['source_hashes'].items(): assert sha256(ROOT/path)==digest
_,selected,_,_=load_data()
groups=['Control','ASD']; levels=[20.,40.]; absolute=[.1,.3,.65,1.3,2.]
baseline=ROOT/'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay300ms'
level_colors={20.:'#1f77b4',40.:'#ff7f0e'}
group_colors={'Control':'#1976d2','ASD':'#d62728'}
times=np.linspace(0,1.8,1801)
edges=np.linspace(0,1.8,91)
summaries=[]; quality=[]; benchmarks=[]; condition_quality=[]
histograms=[]; mixtures=[]; condition_curves=[]; boundary=[]; strongest=[]
loaded={}
for group in groups:
    folder=out/group.lower()
    sub,conditions,data,_,_=prepare_group(selected,group,Path(config['initial_root']))
    posterior=dict(np.load(folder/'posterior_samples.npz'))
    matrix=sample_matrix(posterior)
    assert matrix.shape==(10000,61)
    params={k:jnp.asarray(v) for k,v in vector_params(matrix.mean(0)).items()}
    summary=pd.read_csv(folder/'posterior_summary.csv',float_precision='round_trip')
    assert summary.parameter.tolist()==PARAMETER_NAMES
    summary.insert(0,'Group',group)
    summary['block']=[n.rsplit('_',1)[0] if n!='w' else 'w' for n in summary.parameter]
    summary['condition_index']=list(range(20))*3+[np.nan]
    summary=summary.merge(conditions[['condition_index','ASL_cd','signed_delta']],on='condition_index',how='left',validate='many_to_one')
    summaries.append(summary)
    fitted,score=assess(params,data,sub,conditions,'Gamma/Omega/delay SVI means — 500 ms prior',group)
    score['free_parameters']=61
    quality.append(score); condition_quality.append(fitted)
    report=json.loads((folder/'convergence_summary.json').read_text())
    previous_samples=dict(np.load(baseline/group.lower()/'posterior_samples.npz'))
    previous_params=vector_params(sample_matrix(previous_samples).mean(0))
    _,previous_score=assess(previous_params,data,sub,conditions,'Gamma/Omega/delay SVI means — 300 ms prior',group)
    previous_quality=pd.read_csv(baseline/'fit_quality.csv')
    np.testing.assert_allclose(previous_score['joint_log_likelihood'],
        previous_quality.loc[previous_quality.Group.eq(group),'joint_log_likelihood'].iloc[0],rtol=0,atol=1e-7)
    benchmarks.append(dict(previous_score,free_parameters=61,source=str(baseline/group.lower()/'posterior_samples.npz')))
    for old_name,old_file,old_key,n_parameters in [
        ('TIED joint MLE','joint_fits.json','tied',5),
        ('Shared-scale DDM joint MLE','shared_scale_fits.json','joint',23),
        ('Free-scale DDM shared-delay joint MLE','joint_fits.json','free_ddm',42),
    ]:
        original=json.loads((Path(config['initial_root'])/group.lower()/old_file).read_text())
        x=np.asarray(original[old_key]['x'])
        if n_parameters==5:
            reference=json.loads((Path(config['initial_root'])/group.lower()/'fits.json').read_text())['reference_B']
            g,o,w,tau=tied_coordinates(jnp.asarray(x),data['condition_B'],data['condition_D'],reference)
            old=dict(gamma=g,omega=o,t_E_aff=jnp.repeat(tau,20),w=w)
        elif n_parameters==23:
            old=dict(gamma=np.sinh(x[:20]),omega=np.repeat(np.exp(x[20]),20),t_E_aff=np.repeat(x[22],20),w=x[21])
        else:
            old=dict(gamma=np.sinh(x[:20]),omega=np.exp(x[20:40]),t_E_aff=np.repeat(x[41],20),w=x[40])
        _,old_score=assess(old,data,sub,conditions,old_name,group)
        np.testing.assert_allclose(old_score['joint_log_likelihood'],-original[old_key]['nll'],rtol=1e-10,atol=1e-7)
        benchmarks.append(dict(old_score,free_parameters=n_parameters,source=str(Path(config['initial_root'])/group.lower()/old_file)))
    benchmarks.append(score)
    g,o,tau,w=(params[n] for n in PARAMETER_BLOCKS)
    log_mass=ddm_log_cdf(1.8,g,o,w,tau)
    right_density=np.asarray(jnp.exp(ddm_log_density(times[:,None],g,o,w,tau,side=1)-log_mass))
    left_density=np.asarray(jnp.exp(ddm_log_density(times[:,None],g,o,w,tau,side=-1)-log_mass))
    density=right_density+left_density
    cdf=np.asarray(jnp.exp(ddm_log_cdf(times[:,None],g,o,w,tau)-log_mass))
    for c in conditions.itertuples():
        i=c.condition_index
        condition_curves.append(pd.DataFrame(dict(Group=group,condition_index=i,ASL_cd=c.ASL_cd,
            signed_delta=c.signed_delta,time_s=times,right_density=right_density[:,i],
            left_density=left_density[:,i],density=density[:,i],cdf=cdf[:,i])))
    for level in levels:
        for delta in absolute:
            choose=sub.ASL_cd.eq(level)&np.isclose(sub.signed_delta.abs(),delta)
            obs=sub.loc[choose,'RT_s'].to_numpy()
            ns=sub.loc[choose].groupby('signed_delta').size()
            ids=conditions.loc[conditions.ASL_cd.eq(level)&np.isclose(conditions.signed_delta.abs(),delta),'condition_index'].to_numpy()
            weights=conditions.set_index('condition_index').loc[ids,'n'].to_numpy()/len(obs)
            assert len(ids)==2 and np.isclose(weights.sum(),1)
            curve=density[:,ids]@weights
            cumulative=cdf[:,ids]@weights
            counts,_=np.histogram(obs,bins=edges)
            histograms.append(pd.DataFrame(dict(Group=group,ASL_cd=level,abs_delta=delta,left_s=edges[:-1],
                right_s=edges[1:],count=counts,n=len(obs),density=counts/(len(obs)*.02))))
            mixtures.append(pd.DataFrame(dict(Group=group,ASL_cd=level,abs_delta=delta,time_s=times,
                density=curve,cdf=cumulative,negative_n=int(ns.loc[-delta]),positive_n=int(ns.loc[delta]),
                negative_weight=float(weights[0]),positive_weight=float(weights[1]))))
    for i,name in enumerate(PARAMETER_NAMES):
        if i<20:
            prior=PRIORS['gamma_positive' if conditions.signed_delta.iloc[i]>0 else 'gamma_negative']
        else:
            prior=PRIORS['omega' if i<40 else 't_E_aff' if i<60 else 'w']
        lo,hi=prior['hard']; vals=matrix[:,i]
        causal=float(conditions.guide_delay_upper_s.iloc[i-40]) if 40<=i<60 else hi
        boundary.append(dict(Group=group,parameter=name,hard_low=lo,hard_high=hi,
            fraction_within_1pct_lower_hard=float(np.mean(vals<lo+.01*(hi-lo))),
            fraction_within_1pct_upper_hard=float(np.mean(vals>hi-.01*(hi-lo))),
            effective_upper=causal, fraction_within_1pct_causal_upper=float(np.mean(vals>causal-.01*causal)) if 40<=i<60 else np.nan))
    corr=np.corrcoef(matrix,rowvar=False)
    ia,ib=np.triu_indices(61,1)
    order=np.argsort(-np.abs(corr[ia,ib]))[:20]
    for pos in order:
        strongest.append(dict(Group=group,parameter_1=PARAMETER_NAMES[ia[pos]],parameter_2=PARAMETER_NAMES[ib[pos]],
                              correlation=float(corr[ia[pos],ib[pos]])))
    loaded[group]=dict(summary=summary,conditions=conditions,params=params,posterior=posterior,
                       report=report,sub=sub,fit=fitted,score=score,corr=corr)

tables={
    'parameter_summary.csv':pd.concat(summaries,ignore_index=True), 'fit_quality.csv':pd.DataFrame(quality),
    'benchmark_comparison.csv':pd.DataFrame(benchmarks), 'condition_fit_quality.csv':pd.concat(condition_quality,ignore_index=True),
    'psychometric_comparison.csv':pd.concat(condition_quality,ignore_index=True),
    'rtd_histograms.csv':pd.concat(histograms,ignore_index=True), 'rtd_model_curves.csv':pd.concat(mixtures,ignore_index=True),
    'condition_model_curves.csv':pd.concat(condition_curves,ignore_index=True),
    'posterior_boundary_audit.csv':pd.DataFrame(boundary), 'strongest_correlations.csv':pd.DataFrame(strongest),
}
for name,table in tables.items(): table.to_csv(out/name,index=False,float_format='%.17g')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
hist=tables['rtd_histograms.csv']; curves=tables['rtd_model_curves.csv']
ymax=np.ceil(max(hist.density.max(),curves.density.max())*1.08*2)/2
figure_paths=[]

# %% Two group pages: ASL rows, five absolute differences, psychometric at right.
for group in groups:
    fit=loaded[group]['fit']; info=loaded[group]
    fig=plt.figure(figsize=(21.8,7.6))
    grid=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1,1.7],left=.055,right=.985,bottom=.18,top=.79,hspace=.38,wspace=.33)
    for r,level in enumerate(levels):
        for c,delta in enumerate(absolute):
            ax=fig.add_subplot(grid[r,c])
            h=hist.loc[hist.Group.eq(group)&hist.ASL_cd.eq(level)&hist.abs_delta.eq(delta)]
            m=curves.loc[curves.Group.eq(group)&curves.ASL_cd.eq(level)&curves.abs_delta.eq(delta)]
            ax.stairs(h.density.to_numpy(),np.r_[h.left_s.to_numpy(),h.right_s.iloc[-1]],color=level_colors[level],alpha=.58,lw=1.5)
            ax.plot(m.time_s,m.density,color=level_colors[level],lw=2.1)
            ax.set(xlim=(0,1.8),ylim=(0,ymax),xticks=[0,.6,1.2,1.8])
            ax.set_title(f'|ΔdB| = {delta:g}  ·  n={int(h.n.iloc[0])}',fontsize=10)
            ax.grid(alpha=.16)
            if c==0: ax.set_ylabel(f'ASL {level:g} cd/m²\nDensity (s⁻¹)')
            else: ax.tick_params(labelleft=False)
            if r==1: ax.set_xlabel('RT (s)')
    ax=fig.add_subplot(grid[:,5])
    for level in levels:
        p=fit.loc[fit.ASL_cd.eq(level)].sort_values('signed_delta')
        ax.scatter(p.signed_delta,p.empirical_p_right,color=level_colors[level],s=25,zorder=3)
        ax.plot(p.signed_delta,p.model_p_right,color=level_colors[level],lw=2,label=f'ASL {level:g} cd/m²')
    ax.set(title='Psychometric',xlabel='Signed ΔdB (right − left)',ylabel='P(right)',xlim=(-2.15,2.15),ylim=(-.015,1.015),xticks=[-2,-1,0,1,2])
    ax.grid(alpha=.18); ax.legend(loc='lower right',fontsize=9)
    n_people=info['sub'].Participant.nunique()
    fig.suptitle(f'{group} | human simultaneous RT | {len(info["sub"]):,} trials from {n_people} participants pooled',
                 color=group_colors[group],fontsize=18,fontweight='bold',y=.97)
    fig.text(.5,.915,'Reactive-only, joint RT + choice | 20 independent Gamma–Omega–delay triples + shared bias (61 parameters) | delay prior 0–500 ms',ha='center',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='.5',lw=1.5,alpha=.65,drawstyle='steps-mid',label='Data: 20 ms bins / choice dots'),
                        Line2D([],[],color='.2',lw=2.1,label='Analytic model at arithmetic posterior means')],loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False)
    status='Convergence checks passed' if not info['report']['flagged'] else 'PROVISIONAL: convergence checks not satisfied'
    fig.text(.5,.103,'Condition = RT; 0–1.8 s retained, both choices; raw-trial pooling across participants and sessions; no further exclusion',ha='center',fontsize=10)
    fig.text(.5,.065,'RTDs pool ±Δ within each ASL using retained-trial frequencies; each signed prediction is normalized on 0–1.8 s',ha='center',fontsize=10)
    fig.text(.5,.027,f'{status} | psychometric model points are connected only as guides; no extra sigmoid fit',ha='center',fontsize=10)
    stem=f'human_{group.lower()}_gamma_omega_rtd_psychometric'
    for suffix in ['png','pdf']:
        target=out/f'{stem}.{suffix}';fig.savefig(target,dpi=180);figure_paths.append(target.name)
    plt.close(fig)

# %% A display-only revision can preserve the other published figures byte for byte.
if args.group_figures_only:
    manifest=json.loads((out/'plot_manifest.json').read_text())
    manifest.update(source_sha256=sha256(Path(__file__)),psychometric_data='Dots at observed P(right); no error bars',
                    psychometric_model='Existing conditional probabilities at measured stimuli, joined as visual guides')
    (out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Updated Control and ASD figures; empirical dots without error bars; fitted curves unchanged')
    sys.exit(0)

# %% Gamma, Omega and delay by signed difference, plus the one shared bias.
fig=plt.figure(figsize=(16.7,13.2))
grid=fig.add_gridspec(3,3,width_ratios=[1,1,.32],left=.065,right=.975,bottom=.13,top=.88,wspace=.28,hspace=.32)
ticks=sorted(selected.signed_delta.unique())
for r,(block,label,scale) in enumerate([('gamma','Gamma (dimensionless)',1.),('omega','Omega (s⁻¹)',1.),('t_E_aff','Delay (ms)',1000.)]):
    axes=[]
    for c,level in enumerate(levels):
        ax=fig.add_subplot(grid[r,c],sharey=axes[0] if axes else None);axes.append(ax)
        for group,offset,marker in [('Control',-.018,'o'),('ASD',.018,'*')]:
            p=loaded[group]['summary'];p=p.loc[p.block.eq(block)&p.ASL_cd.eq(level)].sort_values('signed_delta')
            y=p['mean'].to_numpy()*scale
            ax.errorbar(p.signed_delta+offset,y,yerr=np.vstack([y-p.q025.to_numpy()*scale,p.q975.to_numpy()*scale-y]),
                        fmt=marker,color=group_colors[group],ms=4.5 if group=='Control' else 7.,capsize=2,lw=1.1,label=group)
        ax.set(xticks=ticks,xlim=(-2.14,2.14),ylabel=label if c==0 else '')
        ax.tick_params(axis='x',rotation=55,labelsize=8)
        ax.grid(alpha=.18)
        if block=='gamma': ax.axhline(0,color='.4',lw=.8);ax.set_title(f'ASL {level:g} cd/m²',fontsize=14)
        if block=='omega': ax.set_ylim(bottom=0)
        if block=='t_E_aff':
            ax.set_ylim(0,510);ax.set_xlabel('Signed ΔdB (right − left)')
            ax.axhline(300,color='.55',ls=':',lw=.9)
            ax.text(.98,306/510,'Old 300 ms cap',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='.4')
    if r==0: axes[0].legend(frameon=False)
ax=fig.add_subplot(grid[0,2])
bias_low=[];bias_high=[]
for i,group in enumerate(groups):
    p=loaded[group]['summary'].loc[lambda x:x.parameter.eq('w')].iloc[0]
    ax.errorbar(i,p['mean'],yerr=[[p['mean']-p.q025],[p.q975-p['mean']]],fmt='o' if group=='Control' else '*',
                color=group_colors[group],capsize=5,ms=7 if group=='Control' else 10,lw=2)
    bias_low.append(p.q025);bias_high.append(p.q975)
    ax.text(i,p.q975+.007,f'{p["mean"]:.3f}\n[{p.q025:.3f}, {p.q975:.3f}]',ha='center',va='bottom',fontsize=8,color=group_colors[group])
ax.axhline(.5,color='.5',ls='--',lw=1)
ax.set(title='Shared bias w',xticks=[0,1],xticklabels=groups,
       ylim=(min(.4,min(bias_low)-.02),max(.52,max(bias_high)+.04)),xlim=(-.6,1.6),ylabel='Fractional starting position')
ax.tick_params(axis='x',rotation=70);ax.grid(axis='y',alpha=.18)
fig.suptitle('Human RT | Gamma–Omega posterior comparison | delay prior 0–500 ms',y=.97,fontsize=20)
fig.text(.5,.925,'Control blue; ASD red | means and equal-tailed 95% variational credible intervals | 61 parameters per group',ha='center',fontsize=11)
fig.text(.5,.064,'Gamma, Omega and delay vary independently across all 20 signed stimuli; w is shared across stimuli within each group',ha='center',fontsize=10)
fig.text(.5,.036,'Small horizontal offsets separate groups. Pooled-trial variational uncertainty does not represent participant-to-participant variability.',ha='center',fontsize=10)
for suffix in ['png','pdf']:
    target=out/f'human_gamma_omega_parameter_comparison.{suffix}';fig.savefig(target,dpi=180);figure_paths.append(target.name)
plt.close(fig)

# %% Loss confirmation and correlation diagnostics, with all numerical tables saved.
fig,axes=plt.subplots(2,2,figsize=(15,9.5),gridspec_kw={'width_ratios':[1.2,1]})
for r,group in enumerate(groups):
    trace=pd.read_csv(out/group.lower()/'convergence.csv')
    report=loaded[group]['report']
    ax=axes[r,0];ax.plot(trace.step/1000,trace.mean_negative_elbo,color=group_colors[group])
    ax.axvline(report['restored_best_step']/1000,color='.2',ls='--',label=f'Restored {report["restored_best_step"]:,}')
    ax.set(title=f'{group}: window-mean negative ELBO',xlabel='Updates (thousands)',ylabel='Negative ELBO')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    ax.text(.98,.96,f'Follow-up: {report["confirmation_windows"]} windows\nTerminal shift: {report["maximum_terminal_mean_change_sd"]:.3f} SD',
            transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax=axes[r,1];im=ax.imshow(loaded[group]['corr'],vmin=-1,vmax=1,cmap='coolwarm',origin='lower')
    ax.set(title=f'{group}: physical posterior correlations',xticks=[9.5,29.5,49.5,60],xticklabels=['Gamma','Omega','Delay','w'],
           yticks=[9.5,29.5,49.5,60],yticklabels=['Gamma','Omega','Delay','w'])
    for cut in [19.5,39.5,59.5]:ax.axhline(cut,color='white',lw=.6);ax.axvline(cut,color='white',lw=.6)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle('Gamma–Omega SVI convergence and posterior dependence',fontsize=17)
fig.tight_layout(rect=(0,0,1,.96))
for suffix in ['png','pdf']:
    target=out/f'human_gamma_omega_convergence_correlations.{suffix}';fig.savefig(target,dpi=180);figure_paths.append(target.name)
plt.close(fig)
(out/'plot_manifest.json').write_text(json.dumps(dict(figures=figure_paths,rt_bin_s=.02,curve_step_s=.001,
    rt_window_s=[0,1.8],curves='arithmetic posterior means of physical Gamma, Omega, delay and bias',
    empirical_pooling='raw trials; signs pooled only for displayed RTD mixtures',interval='equal-tailed 95% variational credible',
    source_sha256=sha256(Path(__file__)),tables=list(tables)),indent=2))
print(pd.DataFrame(quality).to_string(index=False))
print('Saved',len(figure_paths),'figure files in',out)
