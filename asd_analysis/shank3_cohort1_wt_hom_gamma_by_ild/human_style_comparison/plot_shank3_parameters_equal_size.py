# %% Editable layout settings; reuse saved posterior summaries without refitting
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild'
DEST = FIT/'human_style_comparison'
STEM = 'rat_WT_HOM_all_parameters_equal_size'
COLORS = {'wt':'#4D4D4D', 'hom':'#C24A7A'}
GROUPS = ['wt','hom']
ABLS = [20,40,60]
ILDS = [-16,-8,-4,-2,-1,1,2,4,8,16]
FIGURE_SIZE = (12,8.8)
DPI = 200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# %% Preserve previous figures; match the measured Gamma plotting area exactly
previous = {str(p):sha256(p) for p in DEST.iterdir() if p.is_file()
            and not p.name.startswith(STEM) and p.name!=Path(__file__).name}
manifest = json.loads((DEST/'plot_manifest.json').read_text())
reference = next(panel for figure in manifest['figures'] if figure['kind']=='parameters'
                 for panel in figure['panels'] if panel['panel']=='gamma')
SIDE_INCHES = reference['width_inches']
np.testing.assert_allclose(SIDE_INCHES,reference['height_inches'],atol=1e-12)
params = pd.read_csv(DEST/'parameter_summary.csv',float_precision='round_trip')
average = pd.read_csv(DEST/'rat_WT_HOM_average_omega_summary.csv',float_precision='round_trip')
source_paths = [DEST/'parameter_summary.csv',DEST/'rat_WT_HOM_average_omega_summary.csv',
                DEST/'plot_manifest.json']
source_paths += [FIT/g/'gamma_by_ild/posterior_samples.npz' for g in GROUPS]
source_hashes = {str(p):sha256(p) for p in source_paths}
tables = []
for block in ['gamma','omega','t_E_aff','w','del_go']:
    sub = params.loc[params.block.eq(block)].copy()
    sub['panel'] = block
    sub['category'] = sub.ILD if block=='gamma' else sub.ABL if block=='omega' else sub.group.str.upper()
    tables.append(sub[['panel','group','category','display_mean','display_lower95','display_upper95']])
tables.append(average.assign(panel='average_omega',category=average.group.str.upper()).rename(
    columns={'mean':'display_mean','lower95':'display_lower95','upper95':'display_upper95'})[
        ['panel','group','category','display_mean','display_lower95','display_upper95']])
estimates = pd.concat(tables,ignore_index=True)
assert len(estimates)==34

# Check all 32 parameter intervals and both averaged-Omega intervals against joint draws.
for group in GROUPS:
    with np.load(FIT/group/'gamma_by_ild/posterior_samples.npz') as posterior:
        for row in params.loc[params.group.eq(group)].itertuples():
            i = int(row.parameter.rsplit('_',1)[1]) if row.block=='gamma' else ABLS.index(int(row.ABL)) if row.block=='omega' else 0
            samples = posterior[row.block].reshape(10000,-1)[:,i]
            scale = 1000 if row.block in ['t_E_aff','del_go'] else 1
            expected = np.r_[samples.mean(),np.quantile(samples,[.025,.975])]*scale
            np.testing.assert_allclose([row.display_mean,row.display_lower95,row.display_upper95],expected,atol=1e-12)
        samples = posterior['omega'].mean(axis=1)
        row = average.loc[average.group.eq(group)].iloc[0]
        np.testing.assert_allclose([row['mean'],row.lower95,row.upper95],
            np.r_[samples.mean(),np.quantile(samples,[.025,.975])],atol=1e-12)

# %% Six equally sized square axes, with consistent markers and typography
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig = plt.figure(figsize=FIGURE_SIZE)
panels = [('gamma','Gamma shared across ABLs','Gamma'),
          ('omega','Omega per ABL','Omega (s⁻¹)'),
          ('average_omega','Mean Omega across ABLs','Mean Omega (s⁻¹)'),
          ('t_E_aff','Reactive delay','ms'),
          ('w','Starting bias w','w'),
          ('del_go','del_go','ms')]
geometry, artists = [], []
for index,(block,title,ylabel) in enumerate(panels):
    left = [.9,4.8,8.7][index%3]
    bottom = [5.0,1.2][index//3]
    ax = fig.add_axes([left/FIGURE_SIZE[0],bottom/FIGURE_SIZE[1],
                      SIDE_INCHES/FIGURE_SIZE[0],SIDE_INCHES/FIGURE_SIZE[1]])
    ax.set_gid(block)
    bounds = []
    for group_index,group in enumerate(GROUPS):
        sub = estimates.loc[estimates.panel.eq(block)&estimates.group.eq(group)].copy()
        if block in ['gamma','omega']:
            sub['numeric_category'] = pd.to_numeric(sub.category)
            sub = sub.sort_values('numeric_category')
            x = np.arange(len(sub))+(-.08 if group=='wt' else .08)
        else:
            x = np.array([group_index])
        mean,low,high = [sub[c].to_numpy() for c in ['display_mean','display_lower95','display_upper95']]
        bars = ax.errorbar(x,mean,yerr=np.vstack([mean-low,high-mean]),fmt='o',linestyle='none',
            color=COLORS[group],markersize=4.5,capsize=3,elinewidth=1,capthick=1,
            label=group.upper())
        assert to_hex(bars.lines[0].get_color()).lower()==COLORS[group].lower()
        assert bars.lines[0].get_linestyle()=='None'
        np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:,:,1],
                                   np.column_stack([low,high]),atol=1e-12)
        bounds.extend(np.r_[low,high])
        artists.append(dict(panel=block,group=group,x=x.tolist(),mean=mean.tolist(),
                            lower95=low.tolist(),upper95=high.tolist(),color=COLORS[group]))
    pad = max((max(bounds)-min(bounds))*.15,.002)
    ax.set(title=title,ylabel=ylabel,ylim=(min(bounds)-pad,max(bounds)+pad))
    if block=='gamma':
        ax.set(xticks=np.arange(10),xticklabels=ILDS,xlim=(-.5,9.5),xlabel='Signed ILD (dB)')
        ax.tick_params(axis='x',labelrotation=45)
        ax.axhline(0,color='.6',linewidth=.7)
        ax.legend(frameon=False,fontsize=8)
    elif block=='omega':
        ax.set(xticks=np.arange(3),xticklabels=ABLS,xlim=(-.4,2.4),xlabel='ABL (dB)')
    else:
        ax.set(xticks=[0,1],xticklabels=['WT','HOM'],xlim=(-.5,1.5))
    ax.grid(False,which='both',axis='both')
fig.suptitle('SHANK3 cohort 1 | WT / HOM parameter comparisons',fontsize=16,y=.98)
fig.text(.5,.93,'WT: dark gray   ·   HOM: pink/magenta   |   Posterior means and central 95% credible intervals',
         ha='center',fontsize=10)
fig.text(.5,.055,'All six plotting areas match the original square Gamma panel. Mean Omega averages ABL 20 / 40 / 60 equally within each posterior draw.',
         ha='center',fontsize=9)
fig.text(.5,.025,'Conditional variational uncertainty from saved fits; proactive / lapse parameters fixed. No refitting.',
         ha='center',fontsize=9)
fig.canvas.draw()
for ax in fig.axes:
    position = ax.get_position()
    width,height = position.width*FIGURE_SIZE[0],position.height*FIGURE_SIZE[1]
    np.testing.assert_allclose([width,height],[SIDE_INCHES,SIDE_INCHES],atol=1e-12)
    assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
    geometry.append(dict(panel=ax.get_gid(),width_inches=width,height_inches=height))
assert len(geometry)==6
for ext in ['pdf','png']:
    fig.savefig(DEST/f'{STEM}.{ext}',dpi=DPI,facecolor='white')
plt.close(fig)

# %% Numerical, geometry, source-preservation and PDF audits
estimates.to_csv(DEST/f'{STEM}_estimates.csv',index=False,float_format='%.17g')
for name,expected in {**previous,**source_hashes}.items():
    assert sha256(Path(name))==expected,name
pdfinfo = subprocess.check_output(['pdfinfo',str(DEST/f'{STEM}.pdf')],text=True)
assert int(next(line.split(':')[1] for line in pdfinfo.splitlines() if line.startswith('Pages:')))==1
shutil.copy2(Path(__file__),DEST/Path(__file__).name)
audit = dict(status='passed',source_hashes=source_hashes,previous_exports_unchanged=True,
    parameter_estimates=34,panels=geometry,reference_Gamma_side_inches=SIDE_INCHES,
    grid_visible=False,colors=COLORS,single_page_PDF=True,refitted=False,
    plotted_series=artists,visual_inspection='pending',plotting_source=str(Path(__file__).resolve()),
    plotting_source_sha256=sha256(Path(__file__)))
(DEST/f'{STEM}_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Six square panels, each',SIDE_INCHES,'inches per side')
print('PDF:',DEST/f'{STEM}.pdf')
