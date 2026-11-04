# %% Editable sources, colors, equal square geometry and saved-posterior settings
"""All 37 fitted parameters per genotype, without refitting or new inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_comparison'
FIT_ROOTS = {g:ROOT/f'asd_analysis/shank3_cohort1_{g}_npl_alpha_condition_delay' for g in ['wt','hom']}
COLORS = {'wt':'#000000','hom':'#d62728'}
LABELS = {'wt':'Control (WT)','hom':'ASD (HOM)'}
ABLS, ILDS = [20,40,60], [-16,-8,-4,-2,-1,1,2,4,8,16]
GLOBALS = ['rate_lambda','T_0','theta_E','rate_norm_l','alpha','w','del_go']
FIGURE_SIZE, SIDE_INCHES, DPI = (16.,13.6), 2.8, 200
GENOTYPE_OFFSET = .10
STEM = 'SHANK3_cohort1_WT_HOM_NPL_alpha_parameter_comparison'
OUT.mkdir(exist_ok=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


# %% Frozen inputs and every physical posterior mean/central 95% interval
preserved = {}
rows = []
groups = {}
for group,folder in FIT_ROOTS.items():
    for path in folder.rglob('*'):
        if path.is_file():
            preserved[str(path.relative_to(ROOT))] = sha256(path)
    config = json.loads((folder/'seed_0/configuration.json').read_text())
    run = json.loads((folder/'seed_0/run_summary.json').read_text())
    assert run['status']=='converged' and config['parameter_count']==37
    assert json.loads((folder/'independent_validation.json').read_text())['status']=='passed'
    groups[group] = dict(animals=run['animals'],choices=run['valid_trials'],posterior_draws=run['posterior_draws'],
                         restored_best_step=run['restored_best_step'])
    samples = dict(np.load(folder/'seed_0/posterior_samples.npz'))
    old = pd.read_csv(folder/'seed_0/posterior_summary.csv',float_precision='round_trip').set_index('parameter')
    conditions = pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    assert list(conditions.ABL)==np.repeat(ABLS,10).tolist() and list(conditions.ILD)==ILDS*3
    assert set(samples)==set(GLOBALS+['t_E_aff'])
    for block in GLOBALS+['t_E_aff']:
        array = samples[block]
        assert array.shape==((10000,30) if block=='t_E_aff' else (10000,)) and np.isfinite(array).all()
        vectors = array.reshape(10000,-1)
        for i in range(vectors.shape[1]):
            name = f't_E_aff_{i:02d}' if block=='t_E_aff' else block
            vector = vectors[:,i]
            mean,lo,hi = float(vector.mean()),*np.quantile(vector,[.025,.975])
            np.testing.assert_allclose([mean,lo,hi],old.loc[name,['mean','q025','q975']],atol=1e-12)
            scale = 1000. if block in ['T_0','del_go','t_E_aff'] else 1.
            condition = conditions.iloc[i] if block=='t_E_aff' else None
            assert lo<=mean<=hi
            rows.append(dict(group=group,parameter=name,block=block,condition=i if condition is not None else np.nan,
                ABL=condition.ABL if condition is not None else np.nan,ILD=condition.ILD if condition is not None else np.nan,
                mean=mean,lower95=lo,upper95=hi,display_scale=scale,display_unit='ms' if scale==1000 else 'dimensionless',
                display_mean=scale*mean,display_lower95=scale*lo,display_upper95=scale*hi))
estimates = pd.DataFrame(rows)
assert len(estimates)==74 and estimates.groupby('group').size().to_dict()=={'hom':37,'wt':37}
estimates.to_csv(OUT/'parameter_estimates.csv',index=False,float_format='%.17g')
(OUT/'input_preservation.json').write_text(json.dumps(preserved,indent=2)+'\n')

# %% Seven shared-parameter panels above a one-by-three ABL delay row
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
fig = plt.figure(figsize=FIGURE_SIZE)
layout = [
    ('rate_lambda',r'$\lambda$','Dimensionless',1.,9.2),
    ('T_0',r'$T_0$','Time constant (ms)',4.7,9.2),
    ('theta_E',r'$\theta_E$','Dimensionless',8.4,9.2),
    ('rate_norm_l',r'$\ell$ (rate_norm_l)','Dimensionless',12.1,9.2),
    ('alpha',r'$\alpha$','Dimensionless',2.85,5.35),
    ('w',r'$w$','Dimensionless',6.55,5.35),
    ('del_go','del_go','del_go (ms)',10.25,5.35),
    ('delay_20','Reactive delay | ABL 20 dB','Reactive delay (ms)',2.85,1.5),
    ('delay_40','Reactive delay | ABL 40 dB','Reactive delay (ms)',6.55,1.5),
    ('delay_60','Reactive delay | ABL 60 dB','Reactive delay (ms)',10.25,1.5),
]
delays = estimates.loc[estimates.block.eq('t_E_aff')]
delay_pad = max((delays.display_upper95.max()-delays.display_lower95.min())*.12,1.)
delay_ylim = [float(delays.display_lower95.min()-delay_pad),float(delays.display_upper95.max()+delay_pad)]
geometry, artists = [], []
for panel,title,ylabel,left,bottom in layout:
    ax = fig.add_axes([left/FIGURE_SIZE[0],bottom/FIGURE_SIZE[1],SIDE_INCHES/FIGURE_SIZE[0],SIDE_INCHES/FIGURE_SIZE[1]])
    ax.set_gid(panel)
    is_delay = panel.startswith('delay_')
    endpoints = []
    for group_index,group in enumerate(['wt','hom']):
        if is_delay:
            abl = int(panel.split('_')[1])
            sub = estimates.loc[estimates.group.eq(group)&estimates.block.eq('t_E_aff')&estimates.ABL.eq(abl)].sort_values('ILD')
            assert sub.ILD.tolist()==ILDS
            x = np.arange(10)+(-GENOTYPE_OFFSET if group=='wt' else GENOTYPE_OFFSET)
        else:
            sub = estimates.loc[estimates.group.eq(group)&estimates.block.eq(panel)]
            x = np.array([group_index],dtype=float)
        mean,lo,hi = [sub[c].to_numpy() for c in ['display_mean','display_lower95','display_upper95']]
        bars = ax.errorbar(x,mean,yerr=np.vstack([mean-lo,hi-mean]),fmt='o',linestyle='none',
            color=COLORS[group],markersize=5,capsize=3,elinewidth=1.1,capthick=1.1,label=LABELS[group])
        assert to_hex(bars.lines[0].get_color()).lower()==COLORS[group].lower()
        np.testing.assert_allclose(np.asarray(bars.lines[2][0].get_segments())[:,:,1],np.column_stack([lo,hi]),atol=1e-12)
        endpoints.extend(np.r_[lo,hi])
        artists.append(dict(panel=panel,group=group,parameters=sub.parameter.tolist(),x=x.tolist(),
                            means=mean.tolist(),lower95=lo.tolist(),upper95=hi.tolist(),color=COLORS[group]))
    ax.set(title=title,ylabel=ylabel)
    if is_delay:
        ax.set(xticks=np.arange(10),xticklabels=[str(x) for x in ILDS],xlim=(-.6,9.6),ylim=delay_ylim,xlabel='Signed ILD (dB)')
        ax.tick_params(axis='x',labelrotation=45,labelsize=9)
    else:
        pad = max((max(endpoints)-min(endpoints))*.15,.001)
        ax.set(xticks=[0,1],xticklabels=['Control\n(WT)','ASD\n(HOM)'],xlim=(-.6,1.6),ylim=(min(endpoints)-pad,max(endpoints)+pad))
    ax.grid(False,which='both',axis='both')
fig.suptitle('SHANK3 cohort 1 | NPL+alpha parameter comparison',fontsize=18,y=.987)
fig.legend([Line2D([0],[0],color=COLORS[g],marker='o',linestyle='none') for g in ['wt','hom']],
    [LABELS[g] for g in ['wt','hom']],ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.961),fontsize=12)
fig.text(.5,.918,'Posterior means and central 95% credible intervals | Control: 5 animals, 68,184 choices | ASD: 6 animals, 67,715 choices',ha='center',fontsize=10)
fig.text(.5,.053,'37 fitted parameters per genotype: seven shared parameters and 30 independent signed ABL–ILD delays. All time parameters are shown in ms.',ha='center',fontsize=9)
fig.text(.5,.030,'Intervals describe conditional variational posterior uncertainty with proactive/exponential-lapse parameters fixed; they are not animal SEMs.',ha='center',fontsize=9)
fig.text(.5,.010,'Saved 10,000-draw posteriors; no refitting, condition averaging or sign pooling.',ha='center',fontsize=9)
fig.canvas.draw()
for ax in fig.axes:
    bounds = ax.get_position()
    width,height = bounds.width*FIGURE_SIZE[0],bounds.height*FIGURE_SIZE[1]
    np.testing.assert_allclose([width,height],[SIDE_INCHES,SIDE_INCHES],atol=1e-12)
    assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
    geometry.append(dict(panel=ax.get_gid(),width_inches=width,height_inches=height,
                         xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),square=True))
assert len(fig.axes)==10
for ext in ['png','pdf']:
    fig.savefig(OUT/f'{STEM}.{ext}',dpi=DPI)
plt.close(fig)
for name,digest in preserved.items():
    assert sha256(ROOT/name)==digest,name
manifest = dict(status='generated_pending_independent_verification',parameter_estimates=74,panels=10,
    global_panels=7,delay_panels=3,colors=COLORS,labels=LABELS,groups=groups,posterior_draws=10000,
    delay_ILDs=ILDS,ABLs=ABLS,delay_group_offset=GENOTYPE_OFFSET,geometry=geometry,artists=artists,
    source=str(Path(__file__).relative_to(ROOT)),source_sha256=sha256(Path(__file__)),
    figure_size_inches=list(FIGURE_SIZE),delay_ylim=delay_ylim,
    artifacts={f'{STEM}.{ext}':sha256(OUT/f'{STEM}.{ext}') for ext in ['png','pdf']})
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved one composite PNG/PDF with 10 square panels and 74 posterior estimates:',OUT,flush=True)
