# %% Editable composite settings; initial previews are isolated and never publishable
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR','/tmp/shank3-gamma-by-ild-mpl')
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from shank3_gamma_by_ild_utils import ROOT,OUT,ABLS,ILDS,VARIANTS,LABELS,EXPECTED,sha256

ABS_ILDS=[1,2,4,8,16]
ABL_COLORS={20:'#1f77b4',40:'#ff7f0e',60:'#2ca02c'}
GROUP_COLORS={'wt':'#1f77b4','hom':'#d62728'}
RTD_XLIM=(-.1,.6)
GAMMA_WIDTH_FRACTION=.5  # Compress horizontal spacing without changing height or values.
DPI=180
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initial-preview',action='store_true')
parser.add_argument('--square-gamma',action='store_true',help='Separate parameter figure with an exactly square Gamma panel.')
args=parser.parse_args()
SOURCE=OUT/'initial_preview' if args.initial_preview else OUT
DEST=SOURCE/('square_gamma_figures' if args.square_gamma else 'figures')
DEST.mkdir(exist_ok=True)
parameters=pd.read_csv(SOURCE/'parameter_comparisons.csv',float_precision='round_trip')
points=pd.read_csv(SOURCE/'chronometric_points.csv',float_precision='round_trip')
data={g:pd.read_csv(OUT/g/'rtd_data_bins.csv',float_precision='round_trip') for g in EXPECTED}
models={(g,v):pd.read_csv(SOURCE/g/v/'predictions/pooled_rtd_model_1ms.csv.gz',float_precision='round_trip') for g in EXPECTED for v in VARIANTS}
psych={(g,v):pd.read_csv(SOURCE/g/v/'predictions/psychometric_predictions.csv',float_precision='round_trip') for g in EXPECTED for v in VARIANTS}
RTD_YLIM=(0,float(np.ceil(max(z.density.max() for z in [*data.values(),*models.values()])*1.08)))
CHRONO_YLIM=(0,max(.30,float(np.ceil(max((points.data_mean_RT_s+points.animal_SEM_s).max(),points.model_mean_RT_s.max())*108)/100)))
parameter_limits={}
for block,z in parameters.groupby('block'):
    lo,hi=z.display_lower95.min(),z.display_upper95.max()
    padding=max((hi-lo)*.08,.005 if block=='w' else .02)
    parameter_limits[block]=(float(lo-padding),float(hi+padding))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})
traces=[];figures=[];artifacts={}


def record(artist,kind,**meta):
    item=dict(kind=kind,**meta)
    visible=artist.lines[0] if kind=='errorbar' else artist
    item.update(alpha=1. if visible.get_alpha() is None else visible.get_alpha(),zorder=visible.get_zorder())
    if kind=='stairs':
        values=artist.get_data()
        item.update(edges=values.edges.tolist(),y=values.values.tolist(),linewidth=artist.get_linewidth(),color=to_hex(artist.get_edgecolor()))
    elif kind=='scatter':
        xy=np.asarray(artist.get_offsets())
        item.update(x=xy[:,0].tolist(),y=xy[:,1].tolist(),color=to_hex(artist.get_facecolor()[0]))
    else:
        line=artist.lines[0] if kind=='errorbar' else artist
        x,y=map(lambda a:np.asarray(a,dtype=float),line.get_data())
        item.update(x=x.tolist(),y=y.tolist(),color=to_hex(line.get_color()),linewidth=line.get_linewidth(),linestyle=line.get_linestyle())
        if kind=='errorbar':
            segments=np.asarray(artist.lines[2][0].get_segments())
            item.update(lower=segments[:,0,1].tolist(),upper=segments[:,1,1].tolist())
    traces.append(item)


def save(fig,stem,variant,kind):
    fig.canvas.draw()
    panels=[]
    for ax in fig.axes:
        p=ax.get_position()
        panels.append(dict(panel=ax.get_gid(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            title=ax.get_title(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
            width_inches=p.width*fig.get_figwidth(),height_inches=p.height*fig.get_figheight()))
    assert len(panels)==(5 if kind=='parameters' else 15)
    for ext in ['png','pdf']:
        path=DEST/f'{stem}.{ext}'
        fig.savefig(path,dpi=DPI)
        artifacts[path.name]=sha256(path)
    figures.append(dict(stem=stem,variant=variant,kind=kind,panels=panels))
    plt.close(fig)


# %% Five parameter panels: shared Gamma shown once, with full signed ILDs
for variant in VARIANTS:
    stem=f'SHANK3_cohort1_WT_HOM_{variant}_parameters'
    if args.square_gamma:stem+='_square_gamma'
    fig=plt.figure(figsize=(16,10))
    grid=fig.add_gridspec(2,3,left=.075,right=.975,top=.80,bottom=.12,hspace=.62,wspace=.40)
    table=parameters.loc[parameters.variant.eq(variant)]
    ax=fig.add_subplot(grid[0,:2]);ax.set_gid('gamma')
    for group,offset in [('wt',-.08),('hom',.08)]:
        z=table.loc[table.group.eq(group)&table.block.eq('gamma')].sort_values('ILD')
        y=z.display_mean.to_numpy()
        bars=ax.errorbar(np.arange(10)+offset,y,yerr=np.vstack([y-z.display_lower95,z.display_upper95-y]),
                        fmt='o',ms=4.8,capsize=3,color=GROUP_COLORS[group],linestyle='none',elinewidth=1)
        record(bars,'errorbar',figure=stem,variant=variant,panel='gamma',role='posterior',group=group,ABL=None,block='gamma')
    ax.set(title='Gamma shared across ABL 20, 40 and 60 dB',xticks=np.arange(10),xticklabels=ILDS,
           xlim=(-.5,9.5),ylim=parameter_limits['gamma'],xlabel='Signed ILD (dB)',ylabel='Gamma (dimensionless)')
    ax.tick_params(axis='x',labelrotation=45);ax.grid(axis='y',alpha=.15)
    position=ax.get_position()
    width=position.height*fig.get_figheight()/fig.get_figwidth() if args.square_gamma else position.width*GAMMA_WIDTH_FRACTION
    ax.set_position([position.x0+(position.width-width)/2,position.y0,width,position.height])
    if args.square_gamma:ax.set_title('Gamma shared across ABLs\n20, 40 and 60 dB')
    for row,column,block,label in [(0,2,'omega','Omega (s⁻¹)'),(1,0,'t_E_aff','Reactive delay (ms)'),
                                   (1,1,'w','Starting bias w'),(1,2,'del_go','Readout delay del_go (ms)')]:
        ax=fig.add_subplot(grid[row,column]);ax.set_gid(block)
        by_abl=block=='omega'
        for gi,group in enumerate(['wt','hom']):
            z=table.loc[table.group.eq(group)&table.block.eq(block)].sort_values('ABL')
            x=np.arange(3)+(-.08 if group=='wt' else .08) if by_abl else np.array([gi])
            y=z.display_mean.to_numpy()
            bars=ax.errorbar(x,y,yerr=np.vstack([y-z.display_lower95,z.display_upper95-y]),fmt='o',ms=5,
                            capsize=3,color=GROUP_COLORS[group],linestyle='none',elinewidth=1)
            record(bars,'errorbar',figure=stem,variant=variant,panel=block,role='posterior',group=group,ABL=None,block=block)
        ax.set(title='Omega shared within ABL' if by_abl else 'Shared across all stimulus conditions',
               xticks=np.arange(3) if by_abl else [0,1],xticklabels=ABLS if by_abl else ['WT','HOM'],
               xlim=(-.4,2.4) if by_abl else (-.5,1.5),ylim=parameter_limits[block],ylabel=label,
               xlabel='ABL (dB)' if by_abl else '')
        ax.grid(axis='y',alpha=.15)
    if args.initial_preview:fig.text(.5,.988,'INITIAL VALUES ONLY — NOT FITTED',ha='center',va='top',fontsize=12)
    fig.suptitle('SHANK3 cohort 1 WT–HOM | '+LABELS[variant],fontsize=14,y=.95 if args.initial_preview else .98)
    fig.text(.5,.90,'WT: 5 animals / 68,184 choices; HOM: 6 animals / 67,715 choices\n'
             '16 reactive parameters per genotype; positive and negative ILDs remain independent',ha='center',fontsize=10)
    fig.legend(handles=[Line2D([],[],color=GROUP_COLORS[g],marker='o',linestyle='',label=g.upper()) for g in EXPECTED],
               loc='upper center',bbox_to_anchor=(.5,.87),ncol=2,frameon=False)
    footer='Initial parameter values only; no posterior intervals or fit results.' if args.initial_preview else (
        'Posterior means and central 95% credible intervals from 10,000 draws. Conditional variational uncertainty;\n'
        'genotype-specific proactive/exponential-lapse parameters are frozen. Shared values are displayed once.')
    fig.text(.5,.035,footer,ha='center',fontsize=9)
    save(fig,stem,variant,'parameters')

# %% Four-row behavior composites, preserving empirical pooling and animal SEMs
for variant in ([] if args.square_gamma else VARIANTS):
    stem=f'SHANK3_cohort1_WT_HOM_{variant}_behavior'
    fig=plt.figure(figsize=(20,17))
    rows=fig.add_gridspec(4,1,left=.065,right=.98,top=.89,bottom=.085,height_ratios=[1,1,1.05,1.05],hspace=.65)
    for row,group in enumerate(['wt','hom']):
        grid=rows[row].subgridspec(1,5,wspace=.31)
        for column,ild in enumerate(ABS_ILDS):
            ax=fig.add_subplot(grid[0,column]);ax.set_gid(f'{group}_rtd_{ild}')
            for abl in ABLS:
                z=data[group].loc[data[group].ABL.eq(abl)&data[group].abs_ILD.eq(ild)].sort_values('bin')
                curve=models[group,variant].loc[models[group,variant].ABL.eq(abl)&models[group,variant].abs_ILD.eq(ild)].sort_values('rt_s')
                line,=ax.plot(curve.rt_s,curve.density,color=ABL_COLORS[abl],linestyle='-',linewidth=1.2,alpha=1.,zorder=3)
                record(line,'line',figure=stem,variant=variant,panel=ax.get_gid(),role='rtd_model',group=group,ABL=abl,abs_ILD=ild)
                stairs=ax.stairs(z.density.to_numpy(),np.r_[z.left,z.right.iloc[-1]],color=ABL_COLORS[abl],
                                 linewidth=1.4,alpha=.5,baseline=None,fill=False,zorder=2)
                record(stairs,'stairs',figure=stem,variant=variant,panel=ax.get_gid(),role='rtd_data',group=group,ABL=abl,abs_ILD=ild)
            ax.set(title=f'|ILD| = {ild} dB',xlim=RTD_XLIM,ylim=RTD_YLIM,xlabel='Stimulus-relative RT (s)',xticks=[-.1,0,.2,.4,.6])
            ax.axvline(0,color='.7',linewidth=.6,zorder=0)
            if column==0:ax.set_ylabel('RT density (s⁻¹)')
            else:ax.tick_params(labelleft=False)
            ax.grid(axis='y',alpha=.14)
        e=EXPECTED[group]
        fig.text(.065,rows[row].get_position(fig).y1+.025,
                 f'{"ab"[row]}   {group.upper()} RTDs | {e["animals"]} animals; {e["valid"]:,} valid choices + {e["aborts"]:,} pre-onset aborts',
                 fontsize=12,weight='bold')
    grid=rows[2].subgridspec(1,3,wspace=.55)
    for col,group in enumerate(['wt','hom',None]):
        ax=fig.add_subplot(grid[0,col]);ax.set_gid('chronometric_pooled' if group is None else f'{group}_chronometric')
        ax.set_box_aspect(1)
        curves=[(g,None,GROUP_COLORS[g],offset) for g,offset in [('wt',-.04),('hom',.04)]] if group is None else [
            (group,a,ABL_COLORS[a],offset) for a,offset in zip(ABLS,[-.06,0,.06])]
        for g,abl,color,offset in curves:
            z=points.loc[points.variant.eq(variant)&points.group.eq(g)&points.view.eq('ABL_pooled' if group is None else 'ABL_separate')]
            if abl is not None:z=z.loc[z.ABL.eq(abl)]
            z=z.sort_values('abs_ILD');x=np.arange(5)+offset
            line,=ax.plot(x,z.model_mean_RT_s,color=color,linewidth=1.,zorder=2)
            record(line,'line',figure=stem,variant=variant,panel=ax.get_gid(),role='chronometric_model',group=g,ABL=abl)
            bars=ax.errorbar(x,z.data_mean_RT_s,yerr=z.animal_SEM_s,fmt='o',color=color,ms=4.8,capsize=3,elinewidth=1,linestyle='none',zorder=3)
            record(bars,'errorbar',figure=stem,variant=variant,panel=ax.get_gid(),role='chronometric_data',group=g,ABL=abl)
        ax.set(title='WT vs HOM | ABLs pooled' if group is None else f'{group.upper()} | three ABLs',
               xticks=np.arange(5),xticklabels=ABS_ILDS,xlim=(-.35,4.35),ylim=CHRONO_YLIM,xlabel='|ILD| (dB)',ylabel='Mean RT (s)')
        ax.grid(axis='y',alpha=.14)
        if group is None:
            ax.legend(handles=[Line2D([],[],color=GROUP_COLORS[g],label=g.upper()) for g in EXPECTED],loc='lower right',frameon=False,fontsize=8)
    fig.text(.065,rows[2].get_position(fig).y1+.025,'c   Chronometrics | pooled data means ± SEM across animal means; thin solid model lines',fontsize=12,weight='bold')
    grid=rows[3].subgridspec(1,4,width_ratios=[.6,1,1,.6],wspace=.45)
    for column,group in enumerate(['wt','hom'],start=1):
        ax=fig.add_subplot(grid[0,column]);ax.set_gid(f'{group}_psychometric');ax.set_box_aspect(1)
        for abl in ABLS:
            z=psych[group,variant].loc[psych[group,variant].ABL.eq(abl)].sort_values('ILD')
            line,=ax.plot(z.ILD,z.model_p_right,color=ABL_COLORS[abl],linewidth=1.,zorder=2)
            record(line,'line',figure=stem,variant=variant,panel=ax.get_gid(),role='psychometric_model',group=group,ABL=abl)
            dots=ax.scatter(z.ILD,z.data_p_right,color=ABL_COLORS[abl],s=22,zorder=3)
            record(dots,'scatter',figure=stem,variant=variant,panel=ax.get_gid(),role='psychometric_data',group=group,ABL=abl)
        ax.set(title=f'{group.upper()} | n = {EXPECTED[group]["valid"]:,}',xlim=(-18,18),ylim=(0,1),
               xlabel='Signed ILD (dB)',ylabel='P(right)',xticks=[-16,-8,0,8,16])
        ax.grid(alpha=.14)
    fig.text(.065,rows[3].get_position(fig).y1+.025,'d   Psychometrics | measured points and model predictions at the ten fitted signed ILDs',fontsize=12,weight='bold')
    prefix='INITIAL VALUES ONLY — NOT FITTED\n' if args.initial_preview else ''
    fig.suptitle(prefix+'SHANK3 cohort 1 WT–HOM | '+LABELS[variant],fontsize=16,y=.985)
    fig.legend(handles=[Line2D([],[],color=ABL_COLORS[a],label=f'ABL {a} dB') for a in ABLS]+[
        Line2D([],[],color='.25',linewidth=1.4,alpha=.5,label='RTD data: steps, alpha 0.5'),
        Line2D([],[],color='.25',linewidth=1.2,linestyle='-',label='RTD model: solid')],
        loc='upper center',bbox_to_anchor=(.5,.947),ncol=5,frameon=False)
    fig.text(.5,.025,'Valid choices: success ∈ {1,−1}, finite recorded RT in [0,1] s. RTD display is −0.1 to 0.6 s; '
        '20 ms bins retain normalization over [−1,1] s.\n'
        'RTDs include pre-onset aborts and use exact parent-trial onset/sign weights; no aborts enter valid-trial fitting.\n'
        'Chronometrics and psychometrics normalize each valid-trial onset before averaging; signs/ABLs use observed valid-trial counts.\n'
        'Chronometric dots pool trials; SEM bars weight animal means equally. Proactive and exponential-lapse parameters remain fixed.',
        ha='center',fontsize=8.7)
    save(fig,stem,variant,'behavior')
with gzip.open(DEST/'plotted_series.json.gz','wt') as stream:
    json.dump(traces,stream,allow_nan=False)
manifest=dict(status='initial_preview_only' if args.initial_preview else 'generated_pending_verification',
    initial_preview=args.initial_preview,figures=figures,artifacts=artifacts,parameter_estimates=32,
    RTD_xlim=list(RTD_XLIM),RTD_ylim=list(RTD_YLIM),chronometric_ylim=list(CHRONO_YLIM),parameter_limits=parameter_limits,
    parameter_color=GROUP_COLORS,ABL_colors=ABL_COLORS,source_sha256=sha256(Path(__file__)))
(DEST/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved '+str(len(figures))+' '+('INITIAL-ONLY preview' if args.initial_preview else 'fitted-result')+' composites:',DEST,flush=True)
