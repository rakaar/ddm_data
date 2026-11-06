# %% Saved posterior means and intervals; one Gaussian mu/sigma pair per ABL
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_npl_alpha_gaussian_abl_delay_utils import OUT,GLOBALS,BLOCKS,ABLS,parameter_names,sha256

COLORS={'wt':'#000000','hom':'#d62728'}
FIGSIZE,SIDE,DPI=(12.,12.3),2.8,180
STEM='SHANK3_cohort1_WT_HOM_NPL_alpha_gaussian_ABL_delay_parameter_comparison'
rows,moments=[],[]
for group in ['wt','hom']:
    samples=dict(np.load(OUT/group/'seed_0/posterior_samples.npz'))
    assert len(parameter_names())==13
    for block in BLOCKS:
        vectors=samples[block].reshape(10000,-1)
        for i in range(vectors.shape[1]):
            x=vectors[:,i]
            lo,hi=np.quantile(x,[.025,.975])
            scale=1000 if block in ['T_0','del_go','delay_mu','delay_sigma'] else 1
            rows.append(dict(group=group,block=block,
                parameter=block if block in GLOBALS else f'{block}_ABL{ABLS[i]}',
                abl_index=i if block.startswith('delay_') else np.nan,
                ABL=ABLS[i] if block.startswith('delay_') else np.nan,
                mean=x.mean(),lower95=lo,upper95=hi,display_scale=scale,
                display_mean=x.mean()*scale,display_lower95=lo*scale,display_upper95=hi*scale))
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    m,var=truncnorm.stats(-mu/sd,np.inf,loc=mu,scale=sd,moments='mv')
    for i,abl in enumerate(ABLS):
        for quantity,x in [('truncated_mean',m[:,i]),('truncated_sd',np.sqrt(var[:,i]))]:
            lo,hi=np.quantile(x,[.025,.975])
            moments.append(dict(group=group,abl_index=i,ABL=abl,quantity=quantity,
                posterior_mean_s=x.mean(),lower95_s=lo,upper95_s=hi,
                population='zero-truncated delay before RT-window selection; shared across all signed ILDs'))
estimates=pd.DataFrame(rows)
assert len(estimates)==26
estimates.to_csv(OUT/'parameter_estimates.csv',index=False,float_format='%.17g')
pd.DataFrame(moments).to_csv(OUT/'actual_truncated_delay_moments.csv',index=False,float_format='%.17g')

# %% Nine equal 2.8-inch square axes, without grids
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':False,'pdf.fonttype':42})
fig=plt.figure(figsize=FIGSIZE)
titles=[r'$\lambda$',r'$T_0$',r'$\theta_E$',r'$\ell$',r'$\alpha$','w','del_go',
        'Gaussian μ by ABL','Gaussian σ by ABL']
geometry,artists=[],[]
for panel,(block,title) in enumerate(zip(BLOCKS,titles)):
    r,c=divmod(panel,3)
    ax=fig.add_axes([(1.+3.7*c)/FIGSIZE[0],(8.-3.5*r)/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]])
    ends=[]
    for gi,group in enumerate(['wt','hom']):
        z=estimates.loc[estimates.group.eq(group)&estimates.block.eq(block)].sort_values('ABL')
        x=np.arange(3)+(-.08 if group=='wt' else .08) if block.startswith('delay_') else np.array([gi])
        m,l,h=[z[k].to_numpy() for k in ['display_mean','display_lower95','display_upper95']]
        bars=ax.errorbar(x,m,yerr=np.vstack([m-l,h-m]),fmt='o',color=COLORS[group],
            markersize=5,capsize=3,elinewidth=1.1)
        np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:,:,1],np.c_[l,h],atol=1e-12)
        artists.append(dict(panel=block,group=group,parameters=z.parameter.tolist(),
            x=x.tolist(),mean=m.tolist(),lower95=l.tolist(),upper95=h.tolist(),color=COLORS[group]))
        ends.extend(np.r_[l,h])
    pad=max(np.ptp(ends)*.15,.001)
    ax.set(title=title,ylim=(min(ends)-pad,max(ends)+pad),
        ylabel='Time (ms)' if block in ['T_0','del_go','delay_mu','delay_sigma'] else 'Dimensionless')
    if block.startswith('delay_'):
        ax.set(xticks=[0,1,2],xticklabels=ABLS,xlim=(-.4,2.4),xlabel='ABL (dB)')
    else:
        ax.set(xticks=[0,1],xticklabels=['Control\n(WT)','ASD\n(HOM)'],xlim=(-.6,1.6))
    size=ax.get_position().size*np.asarray(FIGSIZE)
    np.testing.assert_allclose(size,[SIDE,SIDE],atol=1e-12)
    geometry.append(dict(panel=block,width_inches=float(size[0]),height_inches=float(size[1]),ylim=list(ax.get_ylim())))
fig.suptitle('SHANK3 cohort 1 | NPL+alpha | Gaussian delays shared across ILDs',y=.985,fontsize=16)
fig.legend([Line2D([0],[0],marker='o',color=COLORS[g],ls='none') for g in ['wt','hom']],
    ['Control (WT)','ASD (HOM)'],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.953))
fig.text(.5,.913,'13 parameters per genotype; posterior means and central 95% credible intervals from 10,000 draws.',ha='center',fontsize=10)
fig.text(.5,.008,'μ/σ describe the underlying Gaussian, truncated below zero with no upper cutoff. Times in ms.\n'
    'Conditional variational uncertainty with proactive/exponential lapses fixed; intervals are not animal SEMs.',ha='center',fontsize=9)
for suffix in ['png','pdf']:
    fig.savefig(OUT/f'{STEM}.{suffix}',dpi=DPI)
plt.close(fig)
(OUT/'parameter_plot_manifest.json').write_text(json.dumps(dict(status='generated_pending_verification',
    estimates=26,panels=9,colors=COLORS,geometry=geometry,artists=artists,
    artifacts={f'{STEM}.{s}':sha256(OUT/f'{STEM}.{s}') for s in ['png','pdf']},
    source_sha256=sha256(Path(__file__))),indent=2)+'\n')
print('Saved 26 estimates in nine equal square panels.',flush=True)
