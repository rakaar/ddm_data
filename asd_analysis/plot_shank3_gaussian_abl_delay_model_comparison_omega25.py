# %% Editable sources and compact equal-square comparison layout
"""Read saved posteriors only: underlying Gaussian mu/sigma, not truncated moments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'condition_gamma_omega':ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25',
    'npl_alpha':ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'}
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_gaussian_delay_model_comparison_omega25'
ABLS=[20,40,60]
GROUPS=['wt','hom']
COUNTS={'wt':(5,68184),'hom':(6,67715)}
LABELS={'condition_gamma_omega':'Condition-specific Γ/Ω (Ω ≤ 25)','npl_alpha':'NPL+alpha (13 parameters)'}
COLORS={'condition_gamma_omega':'#000000','npl_alpha':'#1f77b4'}
MARKERS={'condition_gamma_omega':'o','npl_alpha':'s'}
FIGSIZE,SIDE,DPI=(8.8,8.8),2.8,200
STEM='SHANK3_cohort1_WT_HOM_gaussian_ABL_delay_model_comparison_omega25_2x2'
OUT.mkdir(exist_ok=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


# %% Reconstruct 24 posterior means and central 95% credible intervals
rows=[];hashes={};source_runs=[]
for model,folder in SOURCES.items():
    print('Reading',model,folder,flush=True)
    for group in GROUPS:
        base=folder/group
        for name in ['fit_trials.csv.gz','parent_trials.csv.gz','conditions.csv','membership.csv','fixed_proactive.json',
                     'preparation.json','seed_0/configuration.json','seed_0/run_summary.json',
                     'seed_0/posterior_samples.npz','seed_0/posterior_summary.csv','seed_0/losses.npy',
                     'seed_0/best_checkpoint.pkl','seed_0/latest_checkpoint.pkl']:
            hashes[str((base/name).relative_to(ROOT))]=sha256(base/name)
        run=json.loads((base/'seed_0/run_summary.json').read_text())
        assert run['status']=='converged' and run['parameter_count']==(68 if model=='condition_gamma_omega' else 13)
        assert (run['animals'],run['valid_trials'])==COUNTS[group]
        source_runs.append(dict(model=model,group=group,status=run['status'],free_parameters=run['parameter_count'],
            restored_best_step=run['restored_best_step'],final_checked_step=run['final_checked_step']))
        samples=np.load(base/'seed_0/posterior_samples.npz')
        reference=pd.read_csv(folder/'parameter_estimates.csv',float_precision='round_trip')
        for block in ['delay_mu','delay_sigma']:
            values=samples[block]
            assert values.shape==(10000,3) and np.isfinite(values).all()
            for i,abl in enumerate(ABLS):
                x=values[:,i]
                low,high=np.quantile(x,[.025,.975])
                mean=x.mean()
                old=reference.loc[reference.group.eq(group)&reference.block.eq(block)&reference.ABL.eq(abl)].iloc[0]
                np.testing.assert_allclose([mean,low,high],[old['mean'],old.lower95,old.upper95],atol=1e-14)
                rows.append(dict(model=model,group=group,block=block,ABL=abl,draws=len(x),animals=COUNTS[group][0],
                    valid_trials=COUNTS[group][1],mean_s=mean,lower95_s=low,upper95_s=high,
                    mean_ms=1000*mean,lower95_ms=1000*low,upper95_ms=1000*high,
                    interpretation='underlying Gaussian before zero truncation and RT-window conditioning'))
        samples.close()
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'delay_posterior_comparison.csv',index=False,float_format='%.17g')
differences=[]
for (group,block,abl),z in summary.groupby(['group','block','ABL']):
    direct=z.loc[z.model.eq('condition_gamma_omega')].iloc[0]
    npl=z.loc[z.model.eq('npl_alpha')].iloc[0]
    differences.append(dict(group=group,block=block,ABL=abl,direct_mean_ms=direct.mean_ms,npl_mean_ms=npl.mean_ms,
        direct_minus_npl_mean_ms=direct.mean_ms-npl.mean_ms))
pd.DataFrame(differences).to_csv(OUT/'delay_posterior_mean_differences.csv',index=False,float_format='%.17g')

# %% WT top / HOM bottom; mu left / sigma right; model-colored dots and caps
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=FIGSIZE)
geometry,artists=[],[]
for row,group in enumerate(GROUPS):
    for col,block in enumerate(['delay_mu','delay_sigma']):
        panel=f'{group}_{block}'
        ax=fig.add_axes([(1.+4.*col)/FIGSIZE[0],(4.7-3.7*row)/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]])
        # Common scale for WT/HOM within each parameter column, covering every CI.
        full=summary.loc[summary.block.eq(block)]
        endpoints=full[['lower95_ms','upper95_ms']].to_numpy().ravel()
        pad=max(np.ptp(endpoints)*.12,.25)
        for index,model in enumerate(SOURCES):
            z=summary.loc[summary.group.eq(group)&summary.block.eq(block)&summary.model.eq(model)].sort_values('ABL')
            x=np.arange(3)+(-.07 if index==0 else .07)
            mean,low,high=[z[k].to_numpy() for k in ['mean_ms','lower95_ms','upper95_ms']]
            plotted=ax.errorbar(x,mean,yerr=np.vstack([mean-low,high-mean]),fmt=MARKERS[model],color=COLORS[model],
                linestyle='none',markersize=5,capsize=3,elinewidth=1.1)
            np.testing.assert_allclose(np.asarray(plotted.lines[2][0].get_segments())[:,:,1],np.c_[low,high],atol=1e-12)
            artists.append(dict(panel=panel,model=model,color=COLORS[model],marker=MARKERS[model],
                ABL=z.ABL.tolist(),x=x.tolist(),mean_ms=mean.tolist(),lower95_ms=low.tolist(),upper95_ms=high.tolist()))
        symbol='μ' if block=='delay_mu' else 'σ'
        ax.set(title=f'{group.upper()} | Gaussian '+('mean μ' if block=='delay_mu' else 'SD σ'),
            xticks=np.arange(3),xticklabels=ABLS,xlim=(-.4,2.4),xlabel='ABL (dB)',ylabel=f'{symbol} (ms)',
            ylim=(min(endpoints)-pad,max(endpoints)+pad))
        size=ax.get_position().size*np.asarray(FIGSIZE)
        np.testing.assert_allclose(size,[SIDE,SIDE],atol=1e-12)
        geometry.append(dict(panel=panel,width_inches=float(size[0]),height_inches=float(size[1]),ylim=list(ax.get_ylim())))
fig.suptitle('SHANK3 cohort 1 | Gaussian sensory-delay comparison | Ω upper 25 s⁻¹',y=.978,fontsize=15)
fig.legend([Line2D([0],[0],marker=MARKERS[m],color=COLORS[m],ls='none',markersize=5) for m in SOURCES],
    [LABELS[m] for m in SOURCES],loc='upper center',bbox_to_anchor=(.5,.943),ncol=2,frameon=False,fontsize=9.5)
fig.text(.5,.892,'WT: 5 animals, 68,184 choices  |  HOM: 6 animals, 67,715 choices',ha='center',fontsize=9)
fig.text(.5,.025,'Posterior means and central 95% credible intervals from 10,000 draws per fit.\n'
    'μ/σ describe the underlying Gaussian before zero truncation; delays are shared across ILDs within ABL.\n'
    'Proactive/exponential lapses fixed; conditional variational uncertainty, not animal SEM.',ha='center',fontsize=8)
for suffix in ['png','pdf']: fig.savefig(OUT/f'{STEM}.{suffix}',dpi=DPI)
plt.close(fig)
manifest=dict(status='generated_pending_verification',source_roots={m:str(p.relative_to(ROOT)) for m,p in SOURCES.items()},
    source_runs=source_runs,source_hashes=hashes,plotted_estimates=24,panels=4,groups=GROUPS,ABLs=ABLS,
    blocks=['delay_mu','delay_sigma'],colors=COLORS,markers=MARKERS,geometry=geometry,artists=artists,
    units='ms',interval='central 2.5th–97.5th percentiles of variational posterior draws',
    interpretation='underlying Gaussian mu/sigma; no truncation or RT-window moment transformation',
    artifacts={f'{STEM}.{s}':sha256(OUT/f'{STEM}.{s}') for s in ['png','pdf']},
    source_sha256=sha256(Path(__file__)))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved 2x2 PNG/PDF and 24-estimate audit:',OUT,flush=True)
