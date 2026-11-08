# %% Editable display settings; saved posteriors only, no refitting
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_gamma_omega_gaussian_abl_delay_utils import OUT,BLOCKS,GLOBALS,ABLS,ILDS,parameter_names,sha256,PRIORS,REFERENCES

COLORS={'wt':'#000000','hom':'#d62728'}
FIGSIZE,SIDE,DPI=(12.,16.),2.8,180
STEM='SHANK3_cohort1_WT_HOM_Gamma_Omega_gaussian_ABL_delay_parameter_comparison'
rows,moments,boundaries,reference_ranges=[],[],[],[]
for group in ['wt','hom']:
    samples=dict(np.load(OUT/group/'seed_0/posterior_samples.npz'))
    names=iter(parameter_names())
    for block in BLOCKS:
        vectors=samples[block].reshape(10000,-1)
        for i in range(vectors.shape[1]):
            x=vectors[:,i]
            lo,hi=np.quantile(x,[.025,.975])
            scale=1000 if block in ['del_go','delay_mu','delay_sigma'] else 1
            name=next(names)
            lo_hard,_,_,hi_hard=np.asarray(PRIORS[block])
            lower=float(lo_hard[i]) if lo_hard.ndim else float(lo_hard)
            upper=float(hi_hard[i]) if hi_hard.ndim else float(hi_hard)
            epsilon=.01*(upper-lower)
            boundaries.append(dict(group=group,parameter=name,hard_lower=lower,hard_upper=upper,
                near_boundary_definition='within 1% of full hard-bound range',
                fraction_near_lower=np.mean(x<lower+epsilon),fraction_near_upper=np.mean(x>upper-epsilon)))
            rows.append(dict(group=group,block=block,parameter=name,index=i,
                ABL=ABLS[i//10] if block in ['gamma','omega'] else ABLS[i] if block.startswith('delay_') else np.nan,
                ILD=ILDS[i%10] if block in ['gamma','omega'] else np.nan,
                mean=x.mean(),lower95=lo,upper95=hi,display_scale=scale,
                display_mean=x.mean()*scale,display_lower95=lo*scale,display_upper95=hi*scale))
    previous=pd.read_csv(REFERENCES[group]/'condition_parameter_means.csv')
    for block in ['gamma','omega']:
        reference_ranges.append(dict(group=group,reference_model='NPL+alpha; Gaussian delays by ABL',parameter=block,
            minimum=previous[block].min(),maximum=previous[block].max(),
            conditions_above_direct_omega_upper=int(previous[block].gt(15).sum()) if block=='omega' else 0))
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    m,var=truncnorm.stats(-mu/sd,np.inf,loc=mu,scale=sd,moments='mv')
    for i,abl in enumerate(ABLS):
        for quantity,x in [('truncated_mean',m[:,i]),('truncated_sd',np.sqrt(var[:,i]))]:
            lo,hi=np.quantile(x,[.025,.975])
            moments.append(dict(group=group,abl_index=i,ABL=abl,quantity=quantity,
                posterior_mean_s=x.mean(),lower95_s=lo,upper95_s=hi,
                population='zero-truncated sensory delay before RT-window selection; shared across ten signed ILDs'))
estimates=pd.DataFrame(rows)
assert len(estimates)==136
pd.DataFrame(boundaries).to_csv(OUT/'posterior_boundary_audit.csv',index=False,float_format='%.17g')
pd.DataFrame(reference_ranges).to_csv(OUT/'reference_evidence_parameter_ranges.csv',index=False,float_format='%.17g')
estimates.to_csv(OUT/'parameter_estimates.csv',index=False,float_format='%.17g')
pd.DataFrame(moments).to_csv(OUT/'actual_truncated_delay_moments.csv',index=False,float_format='%.17g')

# %% Ten identical physical squares; shared y-limits within Gamma/Omega rows
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42})
fig=plt.figure(figsize=FIGSIZE)
panels=[(block,abl,r,c) for r,block in enumerate(['gamma','omega']) for c,abl in enumerate(ABLS)]
panels += [('delay_mu',None,2,.5),('delay_sigma',None,2,1.5),('w',None,3,.5),('del_go',None,3,1.5)]
geometry,artists=[],[]
for block,abl,r,c in panels:
    panel=block+(f'_ABL{abl}' if abl else '')
    ax=fig.add_axes([(1.+3.7*c)/FIGSIZE[0],(11.8-3.65*r)/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]])
    all_block=estimates.loc[estimates.block.eq(block)]
    ends=all_block[['display_lower95','display_upper95']].to_numpy().ravel()
    pad=max(np.ptp(ends)*.12,.001)
    for gi,group in enumerate(['wt','hom']):
        z=all_block.loc[all_block.group.eq(group)]
        if abl: z=z.loc[z.ABL.eq(abl)]
        z=z.sort_values('index')
        if block in ['gamma','omega']: x=np.arange(10)+(-.10 if gi==0 else .10)
        elif block.startswith('delay_'): x=np.arange(3)+(-.08 if gi==0 else .08)
        else: x=np.array([gi])
        m,l,h=[z[k].to_numpy() for k in ['display_mean','display_lower95','display_upper95']]
        bars=ax.errorbar(x,m,yerr=np.vstack([m-l,h-m]),fmt='o',color=COLORS[group],markersize=4.5,capsize=2.5,elinewidth=1.)
        np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:,:,1],np.c_[l,h],atol=1e-12)
        artists.append(dict(panel=panel,group=group,parameters=z.parameter.tolist(),x=x.tolist(),mean=m.tolist(),lower95=l.tolist(),upper95=h.tolist(),color=COLORS[group]))
    title={'gamma':'Gamma','omega':'Omega','delay_mu':'Gaussian μ by ABL','delay_sigma':'Gaussian σ by ABL','w':'w','del_go':'del_go'}[block]
    if abl: title+=f' | ABL {abl} dB'
    ax.set(title=title,ylim=(min(ends)-pad,max(ends)+pad),ylabel='Time (ms)' if block in ['delay_mu','delay_sigma','del_go'] else 'Omega (s⁻¹)' if block=='omega' else 'Dimensionless')
    if block in ['gamma','omega']:
        ax.set(xticks=np.arange(10),xticklabels=[f'{i:+d}' for i in ILDS],xlim=(-.65,9.65),xlabel='Signed ILD (dB)')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    elif block.startswith('delay_'): ax.set(xticks=[0,1,2],xticklabels=ABLS,xlim=(-.4,2.4),xlabel='ABL (dB)')
    else: ax.set(xticks=[0,1],xticklabels=['Control\n(WT)','ASD\n(HOM)'],xlim=(-.6,1.6))
    size=ax.get_position().size*np.asarray(FIGSIZE)
    np.testing.assert_allclose(size,[SIDE,SIDE],atol=1e-12)
    geometry.append(dict(panel=panel,width_inches=float(size[0]),height_inches=float(size[1]),ylim=list(ax.get_ylim())))
fig.suptitle('SHANK3 cohort 1 | Direct Gamma/Omega | Gaussian sensory delays by ABL',y=.985,fontsize=15)
fig.legend([Line2D([0],[0],marker='o',color=COLORS[g],ls='none') for g in ['wt','hom']],['Control (WT)','ASD (HOM)'],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.962))
fig.text(.5,.932,'68 parameters per genotype; 10,000 posterior draws; means and central 95% credible intervals.',ha='center',fontsize=10)
fig.text(.5,.012,'μ/σ describe the underlying Gaussian, truncated below zero, shared across signed ILDs within ABL.\nConditional variational uncertainty; proactive/exponential lapses fixed. Intervals are not animal SEMs.',ha='center',fontsize=9)
for suffix in ['png','pdf']: fig.savefig(OUT/f'{STEM}.{suffix}',dpi=DPI)
plt.close(fig)
(OUT/'parameter_plot_manifest.json').write_text(json.dumps(dict(status='generated_pending_verification',estimates=136,panels=10,colors=COLORS,geometry=geometry,artists=artists,artifacts={f'{STEM}.{s}':sha256(OUT/f'{STEM}.{s}') for s in ['png','pdf']},source_sha256=sha256(Path(__file__))),indent=2)+'\n')
print('Saved 136 estimates in ten equal square panels.',flush=True)
