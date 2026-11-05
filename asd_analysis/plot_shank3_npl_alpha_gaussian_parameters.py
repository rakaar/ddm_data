# %% Editable posterior sources, equal square geometry and genotype colors
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_npl_alpha_gaussian_utils import OUT, GLOBALS, BLOCKS, COUNTS, sha256, parameter_names

COLORS={'wt':'#000000','hom':'#d62728'}
ABLS,ILDS=[20,40,60],[-16,-8,-4,-2,-1,1,2,4,8,16]
FIGSIZE,SIDE,DPI=(16.,17.45),2.8,180
STEM='SHANK3_cohort1_WT_HOM_NPL_alpha_gaussian_parameter_comparison'
rows,moments=[],[]
for group in ['wt','hom']:
    samples=dict(np.load(OUT/group/'seed_0/posterior_samples.npz'))
    assert len(parameter_names())==67
    for block in BLOCKS:
        vectors=samples[block].reshape(10000,-1)
        for i in range(vectors.shape[1]):
            x=vectors[:,i]
            lo,hi=np.quantile(x,[.025,.975])
            scale=1000 if block in ['T_0','del_go','delay_mu','delay_sigma'] else 1
            rows.append(dict(group=group,block=block,parameter=block if i==0 and block in GLOBALS else f'{block}_{i:02d}',
                condition=i if block.startswith('delay_') else np.nan,
                ABL=ABLS[i//10] if block.startswith('delay_') else np.nan,
                ILD=ILDS[i%10] if block.startswith('delay_') else np.nan,
                mean=x.mean(),lower95=lo,upper95=hi,display_mean=x.mean()*scale,
                display_lower95=lo*scale,display_upper95=hi*scale,display_scale=scale))
    mu,sd=samples['delay_mu'],samples['delay_sigma']
    a=mu/sd
    ratio=np.exp(-a*a/2)/np.sqrt(2*np.pi)/ndtr(a)
    actual_mean=mu+sd*ratio
    actual_sd=sd*np.sqrt(np.maximum(1-a*ratio-ratio**2,0))
    for i in range(30):
        for block,x in [('truncated_mean',actual_mean[:,i]),('truncated_sd',actual_sd[:,i])]:
            lo,hi=np.quantile(x,[.025,.975])
            moments.append(dict(group=group,condition=i,ABL=ABLS[i//10],ILD=ILDS[i%10],quantity=block,
                                posterior_mean_s=x.mean(),lower95_s=lo,upper95_s=hi,
                                population='zero-truncated delay distribution before RT-window selection'))
estimates=pd.DataFrame(rows)
assert len(estimates)==134
estimates.to_csv(OUT/'parameter_estimates.csv',index=False,float_format='%.17g')
pd.DataFrame(moments).to_csv(OUT/'actual_truncated_delay_moments.csv',index=False,float_format='%.17g')

# %% Seven scalar panels, Gaussian mu by ABL, Gaussian sigma by ABL
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':False,'pdf.fonttype':42})
fig=plt.figure(figsize=FIGSIZE)
layout=[('rate_lambda',r'$\lambda$','Dimensionless',1.,13.05),
        ('T_0',r'$T_0$','Time constant (ms)',4.7,13.05),
        ('theta_E',r'$\theta_E$','Dimensionless',8.4,13.05),
        ('rate_norm_l',r'$\ell$','Dimensionless',12.1,13.05),
        ('alpha',r'$\alpha$','Dimensionless',2.85,9.2),
        ('w','w','Dimensionless',6.55,9.2),('del_go','del_go','del_go (ms)',10.25,9.2)]
for block,y,label in [('delay_mu',5.35,'Gaussian μ'),('delay_sigma',1.5,'Gaussian σ')]:
    layout.extend((f'{block}_{abl}',f'{label} | ABL {abl} dB',f'{label} (ms)',x,y) for abl,x in zip(ABLS,[2.85,6.55,10.25]))
limits={}
for block in ['delay_mu','delay_sigma']:
    z=estimates.loc[estimates.block.eq(block)]
    pad=max((z.display_upper95.max()-z.display_lower95.min())*.12,1.)
    limits[block]=[float(z.display_lower95.min()-pad),float(z.display_upper95.max()+pad)]
geometry,artists=[],[]
for name,title,ylabel,left,bottom in layout:
    ax=fig.add_axes([left/FIGSIZE[0],bottom/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]])
    ax.set_gid(name)
    block=name.rsplit('_',1)[0] if name.startswith('delay_') else name
    ends=[]
    for gi,group in enumerate(['wt','hom']):
        z=estimates.loc[estimates.group.eq(group)&estimates.block.eq(block)]
        if block.startswith('delay_'):
            z=z.loc[z.ABL.eq(int(name.rsplit('_',1)[1]))].sort_values('ILD')
            x=np.arange(10)+(-.1 if group=='wt' else .1)
        else: x=np.array([gi])
        m,l,h=[z[c].to_numpy() for c in ['display_mean','display_lower95','display_upper95']]
        bars=ax.errorbar(x,m,yerr=np.vstack([m-l,h-m]),fmt='o',color=COLORS[group],markersize=5,capsize=3,elinewidth=1.1)
        np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:,:,1],np.c_[l,h],atol=1e-12)
        artists.append(dict(panel=name,group=group,parameters=z.parameter.tolist(),x=x.tolist(),mean=m.tolist(),lower95=l.tolist(),upper95=h.tolist(),color=COLORS[group]))
        ends.extend(np.r_[l,h])
    ax.set(title=title,ylabel=ylabel)
    if block.startswith('delay_'):
        ax.set(xticks=np.arange(10),xticklabels=ILDS,xlim=(-.6,9.6),ylim=limits[block],xlabel='Signed ILD (dB)')
        ax.tick_params(axis='x',labelrotation=45,labelsize=9)
    else:
        pad=max(np.ptp(ends)*.15,.001)
        ax.set(xticks=[0,1],xticklabels=['Control\n(WT)','ASD\n(HOM)'],xlim=(-.6,1.6),ylim=(min(ends)-pad,max(ends)+pad))
    ax.grid(False)
    bbox=ax.get_position()
    size=[bbox.width*FIGSIZE[0],bbox.height*FIGSIZE[1]]
    np.testing.assert_allclose(size,[SIDE,SIDE],atol=1e-12)
    geometry.append(dict(panel=name,width_inches=size[0],height_inches=size[1],ylim=list(ax.get_ylim()),square=True))
fig.suptitle('SHANK3 cohort 1 | NPL+alpha with Gaussian condition delays',fontsize=18,y=.99)
fig.legend([Line2D([0],[0],marker='o',color=COLORS[g],linestyle='none') for g in ['wt','hom']],
           ['Control (WT)','ASD (HOM)'],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.968))
fig.text(.5,.943,'Posterior means and central 95% credible intervals | 67 fitted parameters per genotype',ha='center',fontsize=10)
fig.text(.5,.035,'μ/σ describe the underlying Gaussian; trial delays are truncated below zero with no upper cutoff. Time units: ms.',ha='center',fontsize=9)
fig.text(.5,.021,'Conditional variational uncertainty with proactive/exponential lapses fixed; bars are not animal SEMs.',ha='center',fontsize=9)
fig.text(.5,.008,'10,000 posterior draws per genotype; one fresh random seed-0 start. All 13 plotting areas are equal squares.',ha='center',fontsize=9)
for suffix in ['png','pdf']: fig.savefig(OUT/f'{STEM}.{suffix}',dpi=DPI)
plt.close(fig)
(OUT/'parameter_plot_manifest.json').write_text(json.dumps(dict(status='generated_pending_verification',estimates=134,panels=13,
    colors=COLORS,geometry=geometry,artists=artists,limits=limits,
    artifacts={f'{STEM}.{s}':sha256(OUT/f'{STEM}.{s}') for s in ['png','pdf']},source_sha256=sha256(Path(__file__))),indent=2)+'\n')
print('Saved all 134 estimates in 13 equal square panels:',OUT,flush=True)
