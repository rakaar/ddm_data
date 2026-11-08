# %% Editable display settings; use saved rat fits and verified pooling tables
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/shank3-human-style-mpl')
from pathlib import Path
import gzip
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from shank3_gamma_by_ild_utils import ROOT, OUT, ABLS, ILDS, EXPECTED, sha256

DEST = OUT / 'human_style_comparison'
COLORS = {'wt':'#4D4D4D', 'hom':'#C24A7A'}
LABELS = {'wt':'WT (Control)', 'hom':'HOM (ASD)'}
ABS_ILDS = [1,2,4,8,16]
RTD_XLIM = (-.1,.6)
MODEL_WIDTH, DATA_WIDTH, DATA_ALPHA = 1.2, 1.4, .5
BEHAVIOR_SIZE, PARAMETER_SIZE, DPI = (26,6), (14,12), 180
RTD_LAYOUT = '1x5_ABL_pooled_by_abs_ILD'
HUMAN = ROOT / 'docs/assets/results/2026-10-23/human_shared_gaussian_delay'
DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'savefig.facecolor':'white'})

paths = [OUT/'rtd_pdf_exports/data_histograms.csv', OUT/'rtd_pdf_exports/model_curves.csv.gz',
    OUT/'rtd_pdf_exports/panel_audit.csv', OUT/'rtd_pdf_exports/validation.json',
    OUT/'chronometric_points.csv', OUT/'parameter_comparisons.csv', OUT/'signed_condition_means.csv',
    ROOT/'ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py',
    HUMAN/'human_shared_gaussian_delay_pooled_comparison.pdf',
    HUMAN/'human_shared_gaussian_delay_parameter_comparison.pdf']
paths += list((OUT/'figures').glob('*.png')) + list((OUT/'figures').glob('*.pdf'))
paths += list((OUT/'square_gamma_figures').glob('*.png')) + list((OUT/'square_gamma_figures').glob('*.pdf'))
paths += list((OUT/'rtd_pdf_exports').glob('*.pdf'))
for group in EXPECTED:
    folder = OUT/group
    paths += [folder/'fit_trials.csv.gz', folder/'parent_trials.csv.gz', folder/'fixed_proactive.json',
        folder/'gamma_by_ild/posterior_samples.npz', folder/'gamma_by_ild/run_summary.json',
        folder/'gamma_by_ild/predictions/psychometric_predictions.csv',
        folder/'gamma_by_ild/predictions/signed_parent_model_1ms.csv.gz', folder/'conditions.csv']
source_hashes = {str(p.resolve()):sha256(p) for p in paths}
assert json.loads((OUT/'rtd_pdf_exports/validation.json').read_text())['status']=='passed'
hist = pd.read_csv(paths[0],float_precision='round_trip').query('layout == @RTD_LAYOUT').rename(columns={'genotype':'group'})
curves = pd.read_csv(paths[1],float_precision='round_trip').query('layout == @RTD_LAYOUT').rename(columns={'genotype':'group'})
counts = pd.read_csv(paths[2],float_precision='round_trip').query('layout == @RTD_LAYOUT').rename(columns={'genotype':'group'})
chrono = pd.read_csv(OUT/'chronometric_points.csv',float_precision='round_trip').query('view == "ABL_pooled"').copy()
parameters = pd.read_csv(OUT/'parameter_comparisons.csv',float_precision='round_trip')
psych_rows, weight_rows, runs, animal_psych_rows, animal_chrono_rows = [], [], {}, [], []

# %% Pool probabilities with actual fitted-trial frequencies; never average ABLs equally
for group,e in EXPECTED.items():
    folder = OUT/group/'gamma_by_ild'
    runs[group] = json.loads((folder/'run_summary.json').read_text())
    assert runs[group]['parameter_count']==16 and runs[group]['posterior_draws']==10000
    signed = pd.read_csv(folder/'predictions/psychometric_predictions.csv',float_precision='round_trip')
    trials = pd.read_csv(OUT/group/'fit_trials.csv.gz',float_precision='round_trip')
    trials['right_choice'] = trials.response_poke.eq(1)
    trials['abs_ILD'] = trials.ILD.abs()
    animal_psych = trials.groupby(['animal','ILD']).agg(trials=('right_choice','size'),
        right=('right_choice','sum'),p_right=('right_choice','mean')).reset_index()
    animal_chrono = trials.groupby(['animal','abs_ILD']).agg(trials=('timed_rt','size'),
        mean_RT_s=('timed_rt','mean')).reset_index()
    animal_psych_rows.append(animal_psych.assign(group=group))
    animal_chrono_rows.append(animal_chrono.assign(group=group))
    for row in chrono.loc[chrono.group.eq(group)].itertuples():
        values = animal_chrono.loc[animal_chrono.abs_ILD.eq(row.abs_ILD),'mean_RT_s']
        np.testing.assert_allclose(row.animal_SEM_s,values.std(ddof=1)/np.sqrt(len(values)),atol=1e-14)
    assert signed.trials.sum()==e['valid']
    for ild in ILDS:
        sub = signed.loc[signed.ILD.eq(ild)].sort_values('ABL')
        n, right = int(sub.trials.sum()), int(sub.right.sum())
        weights = sub.trials.to_numpy()/n
        animal_probabilities = animal_psych.loc[animal_psych.ILD.eq(ild),'p_right']
        sd = animal_probabilities.std(ddof=1)
        psych_rows.append(dict(group=group, ILD=ild, trials=n, right=right, left=n-right,
            data_p_right=right/n, model_p_right=float(weights@sub.model_p_right.to_numpy()),
            animals=len(animal_probabilities),unweighted_animal_mean_p_right=animal_probabilities.mean(),
            animal_sample_SD=sd,animal_SEM=sd/np.sqrt(len(animal_probabilities))))
        for row,weight in zip(sub.itertuples(),weights):
            weight_rows.append(dict(group=group, ILD=ild, ABL=row.ABL, condition=row.condition,
                trials=row.trials, pooled_trials=n, weight=weight, model_p_right=row.model_p_right))
    assert counts.loc[counts.group.eq(group)].display_trials.sum()==e['valid']+e['aborts']
    assert chrono.loc[chrono.group.eq(group)].n.sum()==e['valid']
psych = pd.DataFrame(psych_rows)
assert len(psych)==20 and len(chrono)==10 and len(parameters)==32
for name,table in [('pooled_rtd_histograms.csv',hist), ('pooled_rtd_model_curves.csv.gz',curves),
    ('pooled_rtd_counts.csv',counts), ('pooled_psychometrics.csv',psych),
    ('psychometric_pooling_weights.csv',pd.DataFrame(weight_rows)),
    ('psychometric_animal_estimates.csv',pd.concat(animal_psych_rows,ignore_index=True)),
    ('chronometric_animal_estimates.csv',pd.concat(animal_chrono_rows,ignore_index=True)),
    ('chronometric_points.csv',chrono), ('parameter_summary.csv',parameters)]:
    table.to_csv(DEST/name,index=False,float_format='%.17g',compression='gzip' if name.endswith('.gz') else None)

traces, figures, artifacts = [], [], {}


def record(artist,kind,**metadata):
    entry = dict(kind=kind,**metadata)
    visible = artist.lines[0] if kind=='errorbar' else artist
    entry['alpha'] = 1. if visible.get_alpha() is None else visible.get_alpha()
    if kind=='stairs':
        v = artist.get_data()
        entry.update(edges=v.edges.tolist(),y=v.values.tolist(),color=to_hex(artist.get_edgecolor()),
                     linewidth=artist.get_linewidth(),zorder=artist.get_zorder())
    elif kind=='scatter':
        xy = np.asarray(artist.get_offsets())
        entry.update(x=xy[:,0].tolist(),y=xy[:,1].tolist(),color=to_hex(artist.get_facecolor()[0]))
    else:
        x,y = [np.asarray(v,dtype=float) for v in visible.get_data()]
        entry.update(x=x.tolist(),y=y.tolist(),color=to_hex(visible.get_color()),
                     linewidth=visible.get_linewidth(),linestyle=visible.get_linestyle())
        if kind=='errorbar':
            segments = np.asarray(artist.lines[2][0].get_segments())
            entry.update(lower=segments[:,0,1].tolist(),upper=segments[:,1,1].tolist())
    traces.append(entry)


def save_figure(fig,stem,kind,close=True):
    for ax in fig.axes:
        ax.grid(False,which='both',axis='both')
    fig.canvas.draw()
    panels = []
    for ax in fig.axes:
        position = ax.get_position()
        panels.append(dict(panel=ax.get_gid(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title=ax.get_title(),
            xticks=ax.get_xticks().tolist(),xticklabels=[t.get_text() for t in ax.get_xticklabels()],
            width_inches=position.width*fig.get_figwidth(),height_inches=position.height*fig.get_figheight(),
            grid_visible=any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())))
    figures.append(dict(stem=stem,kind=kind,size_inches=fig.get_size_inches().tolist(),panels=panels))
    for extension in ['png','pdf']:
        path = DEST/f'{stem}.{extension}'
        fig.savefig(path,dpi=DPI)
        artifacts[path.name] = sha256(path)
    if close:plt.close(fig)


# %% Human-style seven-panel row; two rat groups overlaid in every panel
fig = plt.figure(figsize=BEHAVIOR_SIZE)
grid = fig.add_gridspec(1,7,width_ratios=[1,1,1,1,1,1.55,1.35],
    left=.035,right=.99,bottom=.28,top=.70,wspace=.45)
ymax = np.ceil(max(hist.density.max(),curves.model_density.max())*1.08*2)/2
for column,ild in enumerate(ABS_ILDS):
    ax = fig.add_subplot(grid[0,column]); ax.set_gid(f'rtd_{ild}')
    trial_counts = []
    for group in EXPECTED:
        h = hist.loc[hist.group.eq(group)&hist.abs_ILD.eq(ild)].sort_values('bin')
        model = curves.loc[curves.group.eq(group)&curves.abs_ILD.eq(ild)].sort_values('rt_s')
        step = ax.stairs(h.density.to_numpy(),np.r_[h.left,h.right.iloc[-1]],color=COLORS[group],
            alpha=DATA_ALPHA,linewidth=DATA_WIDTH,baseline=None,fill=False,zorder=2)
        line, = ax.plot(model.rt_s,model.model_density,color=COLORS[group],linewidth=MODEL_WIDTH,zorder=3)
        record(step,'stairs',figure='behavior',panel=ax.get_gid(),group=group,role='rtd_data',abs_ILD=ild)
        record(line,'line',figure='behavior',panel=ax.get_gid(),group=group,role='rtd_model',abs_ILD=ild)
        trial_counts.append(int(h['count'].sum()))
    ax.set(xlim=RTD_XLIM,ylim=(0,ymax),xticks=[-.1,0,.2,.4,.6],xlabel='RT relative to stimulus (s)',
           title=f'|ILD| = {ild} dB\nn: WT {trial_counts[0]:,} · HOM {trial_counts[1]:,}')
    ax.axvline(0,color='.8',linewidth=.6,zorder=0)
    if column==0:ax.set_ylabel('Density (s⁻¹)')
    else:ax.tick_params(labelleft=False)

ax = fig.add_subplot(grid[0,5]); ax.set_gid('psychometric')
for group in EXPECTED:
    table = psych.loc[psych.group.eq(group)].sort_values('ILD')
    line, = ax.plot(table.ILD,table.model_p_right,color=COLORS[group],linewidth=MODEL_WIDTH,label=LABELS[group])
    dots = ax.errorbar(table.ILD,table.data_p_right,yerr=table.animal_SEM,fmt='o',linestyle='none',
        color=COLORS[group],markersize=5,markeredgecolor='white',markeredgewidth=.4,
        capsize=2.5,elinewidth=.9,zorder=3)
    record(line,'line',figure='behavior',panel='psychometric',group=group,role='psychometric_model')
    record(dots,'errorbar',figure='behavior',panel='psychometric',group=group,role='psychometric_data')
ax.set(title='Psychometric\nAll three ABLs pooled',xlabel='Signed ILD (dB)',ylabel='P(right)',
       xlim=(-18,18),ylim=(-.015,1.015),xticks=[-16,-8,0,8,16])
ax.legend(loc='lower right',fontsize=8,frameon=False)

ax = fig.add_subplot(grid[0,6]); ax.set_gid('chronometric')
for group in EXPECTED:
    table = chrono.loc[chrono.group.eq(group)].sort_values('abs_ILD')
    line, = ax.plot(table.abs_ILD,table.model_mean_RT_s,color=COLORS[group],linewidth=MODEL_WIDTH)
    dots = ax.errorbar(table.abs_ILD,table.data_mean_RT_s,yerr=table.animal_SEM_s,fmt='o',linestyle='none',
        color=COLORS[group],markersize=5,markeredgecolor='white',markeredgewidth=.4,
        capsize=2.5,elinewidth=.9,zorder=3)
    record(line,'line',figure='behavior',panel='chronometric',group=group,role='chronometric_model')
    record(dots,'errorbar',figure='behavior',panel='chronometric',group=group,role='chronometric_data')
ax.set(title='Chronometric\nAll three ABLs pooled',xlabel='|ILD| (dB)',ylabel='Mean RT (s)',
       xticks=ABS_ILDS,xticklabels=ABS_ILDS,xlim=(0,17),ylim=(0,.30))
fig.suptitle('Rat RT | SHANK3 cohort 1 | Gamma shared across ABLs; Omega per ABL',fontsize=19,y=.97)
fig.text(.5,.90,'WT (Control): 68,184 fitted choices, 5 animals   |   HOM (ASD): 67,715 fitted choices, 6 animals   |   ABL 20 / 40 / 60 dB pooled for display',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='.45',linewidth=DATA_WIDTH,alpha=DATA_ALPHA,label='Data: 20 ms RT bins'),
    Line2D([],[],color='.45',marker='o',linestyle='none',label='Data: pooled dots ± animal SEM'),
    Line2D([],[],color='.2',linewidth=MODEL_WIDTH,label='Model at posterior means'),
    *[Line2D([],[],color=COLORS[g],linewidth=MODEL_WIDTH,label=LABELS[g]) for g in EXPECTED]],
    loc='upper center',bbox_to_anchor=(.5,.855),ncol=5,frameon=False)
rtd_caption = fig.text(.5,.15,'RTDs: valid choices + pre-onset fixation aborts (WT 81,934; HOM 80,989); normalize over [−1,1] s; display −0.1 to 0.6 s.',ha='center',fontsize=10)
fig.text(.5,.105,'Psychometrics / chronometrics: valid choices, RT [0,1] s. Dots pool trials; bars = sample SD of animal estimates / √animals, after pooling ABLs within animal.',ha='center',fontsize=10)
fig.text(.5,.06,'Ten signed-ILD Gammas; three ABL Omegas; one reactive delay; shared w / del_go. Proactive and exponential-lapse values remain frozen.',ha='center',fontsize=10)
status = ' | '.join(f'{g.upper()}: best {runs[g]["restored_best_step"]:,}; loss confirmation passed' for g in EXPECTED)
fig.text(.5,.015,status+' | Model lines connect measured ILDs only.',ha='center',fontsize=10)
save_figure(fig,'rat_WT_HOM_pooled_comparison','behavior',close=False)
# Display-only alternative: reuse the same artists, counts and full-window normalization.
for ax in fig.axes[:5]:
    ax.set(xlim=(0,.6),xticks=[0,.2,.4,.6])
rtd_caption.set_text('RTDs: valid choices + pre-onset fixation aborts (WT 81,934; HOM 80,989); normalize over [−1,1] s; display 0 to 0.6 s.')
save_figure(fig,'rat_WT_HOM_pooled_comparison_0_to_0p6s','behavior_zoom')

# %% Square Gamma above a 2×2 arrangement of the other rat parameters
fig = plt.figure(figsize=PARAMETER_SIZE)
grid = fig.add_gridspec(3,2,height_ratios=[1.5,1,1],left=.095,right=.95,
    bottom=.11,top=.82,hspace=.65,wspace=.48)
ax = fig.add_subplot(grid[0,:]); ax.set_gid('gamma'); ax.set_box_aspect(1)
for group,shift in [('wt',-.08),('hom',.08)]:
    table = parameters.loc[parameters.group.eq(group)&parameters.block.eq('gamma')].sort_values('ILD')
    mean = table.display_mean.to_numpy()
    bars = ax.errorbar(np.arange(10)+shift,mean,yerr=np.vstack([mean-table.display_lower95,table.display_upper95-mean]),
        fmt='o',color=COLORS[group],markersize=4,capsize=3,elinewidth=1,linestyle='none',label=LABELS[group])
    record(bars,'errorbar',figure='parameters',panel='gamma',group=group,role='posterior',block='gamma')
ax.set(title='Gamma shared across ABLs',ylabel='Gamma',xlabel='Signed ILD (dB)',
       xticks=np.arange(10),xticklabels=ILDS,xlim=(-.5,9.5))
ax.tick_params(axis='x',labelrotation=45); ax.axhline(0,color='.6',linewidth=.7)
ax.legend(frameon=False,fontsize=8)
for index,(block,title,factor) in enumerate([('omega','Omega per ABL (s⁻¹)',1),
    ('t_E_aff','Shared reactive delay (ms)',1000),('w','Starting bias w',1),('del_go','del_go',1000)]):
    ax = fig.add_subplot(grid[1+index//2,index%2]); ax.set_gid(block)
    bounds = []
    for position,group in enumerate(EXPECTED):
        table = parameters.loc[parameters.group.eq(group)&parameters.block.eq(block)].sort_values('ABL')
        x = np.arange(3)+(-.08 if group=='wt' else .08) if block=='omega' else np.array([position])
        mean,low,high = [table[name].to_numpy()*factor for name in ['mean','lower95','upper95']]
        bars = ax.errorbar(x,mean,yerr=np.vstack([mean-low,high-mean]),fmt='o',color=COLORS[group],
            markersize=4.5,capsize=4,elinewidth=1,linestyle='none')
        record(bars,'errorbar',figure='parameters',panel=block,group=group,role='posterior',block=block)
        bounds.extend(np.r_[low,high])
    padding = max((max(bounds)-min(bounds))*.2,.002)
    ax.set(title=title,xticks=np.arange(3) if block=='omega' else [0,1],
        xticklabels=ABLS if block=='omega' else ['WT','HOM'],xlim=(-.4,2.4) if block=='omega' else (-.5,1.5),
        xlabel='ABL (dB)' if block=='omega' else '',ylim=(min(bounds)-padding,max(bounds)+padding))
    if block=='del_go':ax.set_ylabel('ms')
fig.suptitle('Rat RT | SHANK3 cohort 1 | WT / HOM parameter comparison',fontsize=18,y=.975)
fig.text(.5,.935,'WT / Control: dark gray   ·   HOM / ASD: pink/magenta   |   Posterior means and central 95% variational credible intervals',ha='center',fontsize=10)
fig.text(.5,.895,'16 parameters per group | Ten signed-ILD Gammas, three ABL Omegas, one reactive delay, w and del_go',ha='center',fontsize=10)
fig.text(.5,.065,'Proactive / exponential-lapse values are fixed. Intervals are conditional variational uncertainty from 10,000 posterior draws.',ha='center',fontsize=10)
fig.text(.5,.04,'Intervals do not describe animal-to-animal variability. Gamma uses equally spaced signed-ILD categories; both signs remain independent.',ha='center',fontsize=10)
fig.text(.5,.015,status,ha='center',fontsize=10)
save_figure(fig,'rat_WT_HOM_parameter_comparison','parameters')

# %% Reproducible output inventory and source snapshots
with gzip.open(DEST/'plotted_series.json.gz','wt') as stream:json.dump(traces,stream,allow_nan=False)
for path,expected in source_hashes.items():assert sha256(Path(path))==expected,path
manifest = dict(status='generated_pending_verification',fit_root=str(OUT),source=str(Path(__file__).resolve()),
    source_sha256=sha256(Path(__file__)),source_hashes=source_hashes,group_colors=COLORS,group_labels=LABELS,
    figures=figures,artifacts=artifacts,RTD_window_s=[-1,1],RTD_display_s=list(RTD_XLIM),RTD_alternate_display_s=[0,.6],
    rtd_trials={'wt':81934,'hom':80989},valid_trials={'wt':68184,'hom':67715},refitted=False)
(DEST/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for source in [Path(__file__),Path(__file__).with_name('verify_shank3_gamma_by_ild_human_style.py')]:
    if source.exists():shutil.copy2(source,DEST/source.name)
(DEST/'README.txt').write_text('Three single-page rat WT/HOM PDFs styled after the human reference PDFs; all without grids.\n'
    'WT / Control = #4D4D4D; HOM / ASD = #C24A7A. Same saved 16-parameter rat fits; no refitting.\n'
    'Behavior: five ABL-pooled RTDs, psychometrics and chronometrics in a seven-panel row.\n'
    'RTDs include valid choices + pre-onset aborts, normalized on [-1,1] s; displays [-0.1,0.6] and [0,0.6] s.\n'
    'Choice / mean-RT panels use fitted valid trials only, recorded RT in [0,1] s.\n'
    'Dots pool trials; capped SEM bars use sample SD of animal estimates / sqrt(animal count).\n'
    'Each animal pools ABLs first; chronometrics also pool signed ILDs into absolute ILD.\n'
    'Parameters: square Gamma; Omega by ABL, reactive delay, w and del_go; central 95% posterior intervals.\n'
    'Audits preserve counts, condition weights, parameters, actual plotted values and source hashes.\n'
    'Run from repository root: .venv/bin/python asd_analysis/plot_shank3_gamma_by_ild_human_style.py\n')
print('Saved three PDFs and matching PNGs:',DEST,flush=True)
