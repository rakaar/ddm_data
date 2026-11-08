# %% Display-only export from saved predictions; no fit or calculation changes.
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_gaussian_delay_mpl')
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% Editable paths and colors.
ROOT = Path(__file__).resolve().parents[1]
fit = ROOT / 'ASD_Human_Data/pooled_asl_shared_omega_shared_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
out = fit / 'pdf_gray_magenta'
out.mkdir(exist_ok=True)
groups = ['Control', 'ASD']
absolute = [.1, .3, .65, 1.3, 2.]
group_colors = {'Control': '#4D4D4D', 'ASD': '#C24A7A'}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'savefig.facecolor':'white'})
names = ['pooled_rtd_histograms.csv', 'pooled_model_curves.csv', 'pooled_psychometric.csv', 'chronometric_points.csv', 'parameter_summary.csv']
tables = {name: pd.read_csv(fit / name, float_precision='round_trip') for name in names}
loaded = {group: {'summary': pd.read_csv(fit / group.lower() / 'posterior_summary.csv'),
                  'report': json.loads((fit / group.lower() / 'convergence_summary.json').read_text())}
          for group in groups}

# %% Participant SEM after pooling ASLs and sessions within each person.
trial_paths = [fit / group.lower() / 'selected_trials.csv' for group in groups]
trials = pd.concat([pd.read_csv(path, float_precision='round_trip') for path in trial_paths], ignore_index=True)
assert len(trials) == 11344 and trials.source_row.is_unique
assert trials.Condition.eq('RT').all() and trials.Valid.eq(1).all()
assert trials.RT_s.between(0, 1.8).all()
trials['abs_delta'] = trials.signed_delta.abs()
participant_psych = trials.groupby(['Group', 'Participant', 'signed_delta']).agg(
    p_right=('ChoseRight', 'mean'), n=('ChoseRight', 'size')).reset_index()
participant_chrono = trials.groupby(['Group', 'Participant', 'abs_delta']).agg(
    mean_RT_s=('RT_s', 'mean'), n=('RT_s', 'size')).reset_index()
psych_sem = participant_psych.groupby(['Group', 'signed_delta']).agg(
    participant_count=('p_right', 'size'), participant_mean=('p_right', 'mean'),
    participant_sd=('p_right', 'std')).reset_index()
chrono_sem = participant_chrono.groupby(['Group', 'abs_delta']).agg(
    participant_count=('mean_RT_s', 'size'), participant_mean=('mean_RT_s', 'mean'),
    participant_sd=('mean_RT_s', 'std')).reset_index()
for table, keys, saved_name in [(psych_sem, ['Group', 'signed_delta'], 'pooled_psychometric.csv'),
                                (chrono_sem, ['Group', 'abs_delta'], 'chronometric_points.csv')]:
    table['participant_sem'] = table.participant_sd / np.sqrt(table.participant_count)
    assert table.participant_count.eq(table.Group.map({'Control': 11, 'ASD': 8})).all()
    tables[saved_name] = tables[saved_name].merge(table, on=keys, how='left', validate='one_to_one')
    assert tables[saved_name].participant_sem.notna().all()
sem_tables = {'psychometric_participant_values.csv': participant_psych,
              'chronometric_participant_values.csv': participant_chrono,
              'psychometric_participant_sem.csv': tables['pooled_psychometric.csv'],
              'chronometric_participant_sem.csv': tables['chronometric_points.csv']}
for name, table in sem_tables.items():
    table.to_csv(out / name, index=False, float_format='%.17g')

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
        h=hist.loc[hist.Group.eq(group)&np.isclose(hist.abs_delta,delta)]
        m=curves.loc[curves.Group.eq(group)&np.isclose(curves.abs_delta,delta)]
        ax.stairs(h.density.to_numpy(),np.r_[h.left_s,h.right_s.iloc[-1]],color=group_colors[group],alpha=.4,lw=1.25)
        ax.plot(m.time_s,m.density,color=group_colors[group],lw=2.2)
        counts.append(int(h.n.iloc[0]))
    ax.set(xlim=(0,1.8),ylim=(0,ymax),xticks=[0,.6,1.2,1.8],xlabel='RT (s)',
           title=f'|ΔdB| = {delta:g}\nn: Control {counts[0]:,} · ASD {counts[1]:,}')
    ax.grid(False)
    if col==0:ax.set_ylabel('Density (s⁻¹)')
    else:ax.tick_params(labelleft=False)
ax=fig.add_subplot(grid[0,5])
for group in groups:
    p=psych.loc[psych.Group.eq(group)].sort_values('signed_delta')
    ax.plot(p.signed_delta,p.model_p_right,color=group_colors[group],lw=2.2,label=group)
    ax.errorbar(p.signed_delta,p.empirical_p_right,yerr=p.participant_sem,fmt='none',
                ecolor=group_colors[group],elinewidth=1,capsize=2.5,zorder=2)
    ax.scatter(p.signed_delta,p.empirical_p_right,color=group_colors[group],s=28,edgecolor='white',linewidth=.4,zorder=3)
ax.set(title='Psychometric\nBoth ASLs pooled',xlabel='Signed ΔdB (right − left)',ylabel='P(right)',
       xlim=(-2.15,2.15),ylim=(-.015,1.015),xticks=[-2,-1,0,1,2])
ax.set_ylim(min(-.015,float((psych.empirical_p_right-psych.participant_sem).min())-.01),
            max(1.015,float((psych.empirical_p_right+psych.participant_sem).max())+.01))
ax.grid(False);ax.legend(loc='lower right',fontsize=9)
ax=fig.add_subplot(grid[0,6])
for group in groups:
    p=points.loc[points.Group.eq(group)].sort_values('abs_delta')
    ax.plot(p.abs_delta,p.model_mean_RT_s,color=group_colors[group],lw=2.2)
    ax.errorbar(p.abs_delta,p.data_mean_RT_s,yerr=p.participant_sem,fmt='none',
                ecolor=group_colors[group],elinewidth=1,capsize=2.5,zorder=2)
    ax.scatter(p.abs_delta,p.data_mean_RT_s,color=group_colors[group],s=28,edgecolor='white',linewidth=.4,zorder=3)
ax.set(title='Chronometric\nBoth ASLs pooled',xlabel='|ΔdB|',ylabel='Mean RT (s)',
       xticks=absolute,xticklabels=['0.1','0.3','0.65','1.3','2'],xlim=(0,2.1))
v=np.r_[points.model_mean_RT_s,points.data_mean_RT_s-points.participant_sem,
        points.data_mean_RT_s+points.participant_sem]
ax.set_ylim(np.floor((v.min()-.035)*20)/20,np.ceil((v.max()+.035)*20)/20);ax.grid(False)
fig.suptitle('Human simultaneous RT | one shared Gaussian delay per group | ASLs pooled in fitting',fontsize=19,y=.97)
fig.text(.5,.90,'Control: 6,574 trials, 11 participants   |   ASD: 4,770 trials, 8 participants   |   ASL 20 + 40 cd/m²   |   16 parameters per group',ha='center',fontsize=11)
fig.legend(handles=[Line2D([],[],color='.45',lw=1.4,alpha=.6,label='Data: 20 ms RT bins'),
    Line2D([],[],color='.45',marker='o',linestyle='none',label='Data: choice / mean RT ± participant SEM'),
    Line2D([],[],color='.2',lw=2.2,label='Model at posterior means'),
    *[Line2D([],[],color=group_colors[g],lw=2.2,label=g) for g in groups]],
    loc='upper center',bbox_to_anchor=(.5,.855),ncol=5,frameon=False)
fig.text(.5,.15,'Condition = RT; Valid = 1; 0–1.8 s retained | Both choices; raw trials pooled across participants, sessions and ASLs',ha='center',fontsize=10)
fig.text(.5,.105,'Gamma per signed ΔdB | One Omega and Gaussian delay μ/σ across all stimuli within each group | Uniform lapses on 0–1.8 s',ha='center',fontsize=10)
fig.text(.5,.06,'Data bars: participant sample SD / √N (Control N=11; ASD N=8); pooled-trial dots unchanged | Model lines join measured differences',ha='center',fontsize=10)
fig.text(.5,.015,fit_status,ha='center',fontsize=10)
figures=[]
for suffix in ['png','pdf']:
    name=f'human_shared_gaussian_delay_pooled_comparison_gray_magenta.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
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
ax.axhline(0,color='.6',lw=.7);ax.legend(frameon=False);ax.grid(False)
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
    ax.set(title=label,xlim=(-.5,1.5),xticks=[0,1],xticklabels=groups,ylim=(min(bounds)-pad,max(bounds)+pad));ax.grid(False)
fig.suptitle('Human RT | one shared Gaussian delay per group | parameter comparison',fontsize=18,y=.975)
fig.text(.5,.935,'Control dark gray · ASD pink/magenta | Posterior means and equal-tailed 95% variational credible intervals',ha='center',fontsize=11)
fig.text(.5,.895,'16 parameters per group | Gamma per signed ΔdB; Omega, Gaussian μ/σ, bias and lapses shared across stimuli',ha='center',fontsize=10)
fig.text(.5,.065,'μ and σ describe trial-varying delays: Gaussian truncated below zero, with no upper cutoff.',ha='center',fontsize=10)
fig.text(.5,.04,'Pooled-trial variational approximation; intervals do not represent participant-to-participant variability.',ha='center',fontsize=10)
fig.text(.5,.015,fit_status,ha='center',fontsize=10)
for suffix in ['png','pdf']:
    name=f'human_shared_gaussian_delay_parameter_comparison_gray_magenta.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
plt.close(fig)


# %% Record the inputs and palette used for this separate export.
inputs = [fit / name for name in names]
inputs += trial_paths
inputs += [fit / g.lower() / n for g in groups for n in ['posterior_summary.csv', 'convergence_summary.json']]
inputs += [Path(__file__), ROOT / 'ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py']
manifest = {'description': 'Grid-free gray/magenta export with participant SEM; existing fits and data/model centers unchanged',
            'colors': group_colors, 'grid': False,
            'SEM': 'Sample SD (ddof=1) of participant P(right) or mean RT, divided by sqrt(N); ASLs/sessions pooled within person; bars centered on original pooled-trial dots',
            'SEM_bars': {'psychometric': len(psych_sem), 'chronometric': len(chrono_sem)},
            'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'output_sha256': {n: hashlib.sha256((out/n).read_bytes()).hexdigest() for n in figures+list(sem_tables)}}
(out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name in figures:
    print(out / name)
