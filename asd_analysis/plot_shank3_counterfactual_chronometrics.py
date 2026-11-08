# %% Editable display settings; plots use the simulated, not exact, mean RTs
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from shank3_wt_gamma_omega_utils import ROOT,sha256

DEST = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild/counterfactual_chronometrics'
COLORS = {'wt':'#4D4D4D','hom':'#C24A7A'}
ABS_ILDS = [1,2,4,8,16]
SIZE = (12,5)
SIDE = 2.65
DPI = 200
YLIM = (.10,.25)
COMBINED_PDF = 'rat_WT_HOM_delay_and_omega_swap_chronometrics.pdf'
points = pd.read_csv(DEST/'pooled_chronometrics.csv',float_precision='round_trip')
observed = pd.read_csv(DEST/'observed_chronometrics.csv',float_precision='round_trip')
config = json.loads((DEST/'configuration.json').read_text())
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
traces,figures,artifacts = [], [], {}
combined = PdfPages(DEST/COMBINED_PDF)

# %% Recipient color stays fixed: intact solid, donor-parameter swap dashed
for swap,parameter in [('delay_swap','t_E_aff'),('omega_swap','Omega per ABL')]:
    fig = plt.figure(figsize=SIZE)
    axes = [fig.add_axes([x/SIZE[0],1.35/SIZE[1],SIDE/SIZE[0],SIDE/SIZE[1]])
            for x in [.85,4.7,8.55]]
    panels = [('intact_comparison',[('wt','intact'),('hom','intact')]),
              ('wt_swap',[('wt','intact'),('wt',swap)]),
              ('hom_swap',[('hom','intact'),('hom',swap)])]
    for index,(ax,(pid,curves)) in enumerate(zip(axes,panels)):
        ax.set_gid(pid)
        for group,scenario in curves:
            table = points.loc[points.group.eq(group)&points.scenario.eq(scenario)].sort_values('abs_ILD')
            label = group.upper()+' intact' if scenario=='intact' else group.upper()+' with '+('HOM' if group=='wt' else 'WT')+' '+parameter
            line, = ax.plot(table.abs_ILD,table.simulated_mean_RT_s,color=COLORS[group],
                linestyle='-' if scenario=='intact' else '--',linewidth=1.4,label=label)
            traces.append(dict(figure=swap,panel=pid,group=group,scenario=scenario,role='simulated',
                x=line.get_xdata().tolist(),y=line.get_ydata().tolist(),color=line.get_color(),
                linestyle=line.get_linestyle(),linewidth=line.get_linewidth()))
        if index==0:
            for group in ['wt','hom']:
                table = observed.loc[observed.group.eq(group)].sort_values('abs_ILD')
                bars = ax.errorbar(table.abs_ILD,table.data_mean_RT_s,yerr=table.animal_SEM_s,
                    fmt='o',linestyle='none',color=COLORS[group],markersize=4.5,
                    capsize=2.5,elinewidth=.9,markeredgecolor='white',markeredgewidth=.4,zorder=3)
                segments = np.asarray(bars.lines[2][0].get_segments())
                traces.append(dict(figure=swap,panel=pid,group=group,role='observed',
                    x=table.abs_ILD.tolist(),y=table.data_mean_RT_s.tolist(),
                    lower=segments[:,0,1].tolist(),upper=segments[:,1,1].tolist(),color=COLORS[group]))
        ax.set(title=['Intact WT and HOM','WT: '+parameter+' swap','HOM: '+parameter+' swap'][index],
            xlabel='|ILD| (dB)',ylabel='Mean RT (s)',xticks=ABS_ILDS,xlim=(0,17),ylim=YLIM)
        ax.grid(False,which='both',axis='both')
        ax.legend(loc='lower left',frameon=False,fontsize=8)
    fig.suptitle('SHANK3 cohort 1 | '+('Reactive-delay swaps (t_E_aff only)' if swap=='delay_swap' else 'Omega swaps at matching ABLs'),
                 fontsize=16,y=.97)
    fig.text(.5,.88,'WT: dark gray, 5 animals / 68,184 fitted trials   ·   HOM: magenta, 6 animals / 67,715 fitted trials',
             ha='center',fontsize=10)
    fig.text(.5,.16,'Lines: simulated mean RTs; intact solid, swapped dashed. Left-panel dots: observed pooled means ± SEM across animals.',
             ha='center',fontsize=9)
    fig.text(.5,.10,'All ABLs and signed ILDs pooled by recipient fitted-trial counts; RT conditioned on [0,1] s at every exact planned onset.',
             ha='center',fontsize=9)
    fig.text(.5,.04,'Only '+parameter+' is exchanged. Recipient Gamma, bias, del_go, proactive/lapse values and trial design remain fixed; no refitting.',
             ha='center',fontsize=9)
    fig.canvas.draw()
    geometry = []
    for ax in axes:
        p = ax.get_position()
        width,height = p.width*SIZE[0],p.height*SIZE[1]
        np.testing.assert_allclose([width,height],[SIDE,SIDE],atol=1e-12)
        geometry.append(dict(panel=ax.get_gid(),width_inches=width,height_inches=height,
            ylim=list(ax.get_ylim()),xlim=list(ax.get_xlim()),
            grid_visible=any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())))
    stem = 'rat_WT_HOM_'+swap+'_chronometrics'
    for ext in ['pdf','png']:
        path = DEST/f'{stem}.{ext}'
        fig.savefig(path,dpi=DPI,facecolor='white')
        artifacts[path.name] = sha256(path)
    figures.append(dict(swap=swap,stem=stem,panels=geometry))
    combined.savefig(fig,facecolor='white')
    plt.close(fig)
combined.close()
artifacts[COMBINED_PDF] = sha256(DEST/COMBINED_PDF)
(DEST/'plot_manifest.json').write_text(json.dumps(dict(figures=figures,artifacts=artifacts,
    colors=COLORS,plotted_series=traces,combined_PDF=COMBINED_PDF,ylim=list(YLIM),
    source=str(Path(__file__).resolve()),source_sha256=sha256(Path(__file__))),indent=2)+'\n')
shutil.copy2(Path(__file__),DEST/Path(__file__).name)
print('Saved delay-swap and Omega-swap PDFs / PNGs:',DEST,flush=True)
print('Combined PDF:',DEST/COMBINED_PDF,flush=True)
