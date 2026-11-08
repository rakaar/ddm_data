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
out = fit / 'pdf_gray_magenta' / 'square_gamma'
out.mkdir(parents=True, exist_ok=True)
groups = ['Control', 'ASD']
absolute = [.1, .3, .65, 1.3, 2.]
group_colors = {'Control': '#4D4D4D', 'ASD': '#C24A7A'}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'savefig.facecolor':'white'})
names = ['parameter_summary.csv']
tables = {name: pd.read_csv(fit / name) for name in names}
loaded = {group: {'summary': pd.read_csv(fit / group.lower() / 'posterior_summary.csv'),
                  'report': json.loads((fit / group.lower() / 'convergence_summary.json').read_text())}
          for group in groups}

# %% Square Gamma, with the existing six scalar panels.
fit_status=' | '.join(f'{g}: best {loaded[g]["report"]["restored_best_step"]:,}; '+
    ('loss confirmation incomplete' if loaded[g]['report']['flagged'] else 'loss confirmation passed') for g in groups)
figures=[]

fig=plt.figure(figsize=(14,12))
outer=fig.add_gridspec(3,3,height_ratios=[1.65,1,1],left=.08,right=.97,bottom=.11,top=.85,hspace=.65,wspace=.42)
ax=fig.add_subplot(outer[0,:]); ax.set_gid('gamma'); ax.set_box_aspect(1)
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
    ax=fig.add_subplot(outer[1+index//3,index%3]); ax.set_gid(name); bounds=[]
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
    name=f'human_shared_gaussian_delay_parameter_comparison_gray_magenta_square_gamma.{suffix}';fig.savefig(out/name,dpi=180);figures.append(name)
fig.canvas.draw()
panels = {ax.get_gid(): {'width_inches': ax.get_position().width * fig.get_figwidth(),
                        'height_inches': ax.get_position().height * fig.get_figheight()}
          for ax in fig.axes}
assert np.isclose(panels['gamma']['width_inches'], panels['gamma']['height_inches'])
plt.close(fig)


# %% Export provenance; fit and plotted parameter values are unchanged.
inputs = [fit / name for name in names]
inputs += [fit / g.lower() / n for g in groups for n in ['posterior_summary.csv', 'convergence_summary.json']]
inputs += [Path(__file__), ROOT / 'asd_analysis/plot_shank3_gamma_by_ild.py']
manifest = {'description': 'Square Gamma display alternative; original x scale and posterior values preserved',
            'colors': group_colors, 'grid': False, 'panels': panels,
            'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'output_sha256': {n: hashlib.sha256((out/n).read_bytes()).hexdigest() for n in figures}}
(out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(panels, indent=2))
for name in figures:
    print(out / name)
