# %% Editable exports; reuse saved fitted predictions without refitting
from pathlib import Path
import json
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_gamma_by_ild_utils import ROOT, OUT, ABLS, EXPECTED, sha256

DEST = OUT / 'rtd_pdf_exports'
XLIM = (-.1, .6)
ABS_ILDS = [1, 2, 4, 8, 16]
ABL_COLORS = {20:'#1f77b4', 40:'#ff7f0e', 60:'#2ca02c'}
MODEL_COLOR, MODEL_WIDTH, DATA_WIDTH, DATA_ALPHA = 'black', 1.2, 1.4, .5
LAYOUTS = [
    ('3x5_ABL_by_abs_ILD', 3, 5, [(a,i) for a in ABLS for i in ABS_ILDS], (17.5,10.5)),
    ('1x5_ABL_pooled_by_abs_ILD', 1, 5, [(None,i) for i in ABS_ILDS], (17.5,4.3)),
    ('1x3_abs_ILD_pooled_by_ABL', 1, 3, [(a,None) for a in ABLS], (10.5,4.3))]
DEST.mkdir(exist_ok=True)
(DEST / 'previews').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'savefig.facecolor':'white'})
sources = [Path(__file__)]
loaded = {}
for group in EXPECTED:
    folder = OUT / group
    paths = dict(data=folder/'rtd_data_bins.csv', conditions=folder/'conditions.csv',
        parent=folder/'parent_trials.csv.gz', model=folder/'gamma_by_ild/predictions/signed_parent_model_1ms.csv.gz',
        reference=folder/'gamma_by_ild/predictions/pooled_rtd_model_1ms.csv.gz',
        audit=folder/'gamma_by_ild/predictions/rtd_panel_audit.csv',
        model_bins=folder/'gamma_by_ild/predictions/pooled_rtd_model_bins.csv')
    sources.extend(paths.values())
    sources.extend((folder/'gamma_by_ild').glob('*.npz'))
    sources.extend((folder/'gamma_by_ild').glob('*.pkl'))
    loaded[group] = {key:pd.read_csv(path, float_precision='round_trip') for key,path in paths.items()}
hashes = {str(f.relative_to(ROOT)):sha256(f) for f in sources}


def pool_panel(tables, abl, abs_ild):
    """Sum empirical counts; average unnormalized model curves using parent counts."""
    selected = tables['conditions']
    if abl is not None:selected = selected.loc[selected.ABL.eq(abl)]
    if abs_ild is not None:selected = selected.loc[selected.ILD.abs().eq(abs_ild)]
    data = tables['data']
    if abl is not None:data = data.loc[data.ABL.eq(abl)]
    if abs_ild is not None:data = data.loc[data.abs_ILD.eq(abs_ild)]
    first = data.loc[data.ABL.eq(data.ABL.iloc[0]) & data.abs_ILD.eq(data.abs_ILD.iloc[0])].sort_values('bin')
    edges = np.r_[first.left, first.right.iloc[-1]]
    counts = data.groupby('bin')['count'].sum().reindex(range(100)).to_numpy(dtype=int)
    n = int(counts.sum())
    curves = [tables['model'].loc[tables['model'].condition.eq(c)].sort_values('rt_s') for c in selected.condition]
    time = curves[0].rt_s.to_numpy()
    weights = selected.parent_trials.to_numpy()
    raw_density = np.average([z.unnormalized_density for z in curves], axis=0, weights=weights)
    raw_cdf = np.average([z.unconditional_cdf for z in curves], axis=0, weights=weights)
    mass = float(raw_cdf[-1] - raw_cdf[0])
    return dict(conditions=selected.condition.tolist(), time=time, edges=edges, count=counts, n=n,
        parent_n=int(weights.sum()), data_density=counts/(n*np.diff(edges)),
        model_density=raw_density/mass, model_cdf=(raw_cdf-raw_cdf[0])/mass,
        model_bin_mass=np.diff(raw_cdf[::20])/mass, model_window_mass=mass)


# %% Prepare and independently verify the 46 panels before drawing
panels, audits, histogram_rows, curve_rows, model_bin_rows = {}, [], [], [], []
for group, tables in loaded.items():
    parent = tables['parent']
    abort_rt = parent.fix_time/1000 - parent.intended_fix_time/1000
    valid = parent.success.isin([1,-1]) & np.isfinite(parent.timed_rt) & parent.timed_rt.between(0,1)
    abort = parent.abort_type.eq('Fixation') & np.isfinite(abort_rt) & abort_rt.ge(-1) & abort_rt.lt(0)
    assert valid.sum()==EXPECTED[group]['valid'] and abort.sum()==EXPECTED[group]['aborts']
    display = parent.loc[valid|abort].copy()
    display['display_rt'] = np.where(valid.loc[display.index], display.timed_rt, abort_rt.loc[display.index])
    for layout, rows, columns, selections, size in LAYOUTS:
        total = 0
        for panel_index, (abl, ild) in enumerate(selections):
            z = pool_panel(tables, abl, ild)
            panels[group,layout,panel_index] = z
            selected = display
            if abl is not None:selected = selected.loc[selected.ABL_mapped.eq(abl)]
            if ild is not None:selected = selected.loc[selected.ILD.abs().eq(ild)]
            np.testing.assert_array_equal(np.histogram(selected.display_rt, z['edges'])[0], z['count'])
            assert len(selected)==z['n']
            np.testing.assert_allclose(np.diff(z['edges']), .02, atol=1e-14)
            np.testing.assert_allclose(np.diff(z['time']), .001, atol=1e-14)
            np.testing.assert_allclose(np.sum(z['data_density']*np.diff(z['edges'])), 1, atol=1e-12)
            np.testing.assert_allclose(z['model_bin_mass'].sum(), 1, atol=1e-12)
            area = float(trapezoid(z['model_density'], z['time']))
            np.testing.assert_allclose(area, 1, atol=3e-4)
            # A distinct reconstruction starts from the earlier unit-area pair curves.
            ref = tables['audit']
            if abl is not None:ref = ref.loc[ref.ABL.eq(abl)]
            if ild is not None:ref = ref.loc[ref.abs_ILD.eq(ild)]
            expected_density, expected_bins, effective_weights = [], [], []
            for pair in ref.itertuples():
                c = tables['reference'].loc[tables['reference'].ABL.eq(pair.ABL)&tables['reference'].abs_ILD.eq(pair.abs_ILD)].sort_values('rt_s')
                b = tables['model_bins'].loc[tables['model_bins'].ABL.eq(pair.ABL)&tables['model_bins'].abs_ILD.eq(pair.abs_ILD)].sort_values('bin')
                expected_density.append(c.density.to_numpy()); expected_bins.append(b.window_mass.to_numpy())
                effective_weights.append((pair.negative_parent_trials+pair.positive_parent_trials)*pair.model_window_mass)
            np.testing.assert_allclose(z['model_density'], np.average(expected_density,axis=0,weights=effective_weights), atol=1e-12)
            np.testing.assert_allclose(z['model_bin_mass'], np.average(expected_bins,axis=0,weights=effective_weights), atol=1e-12)
            meta = dict(genotype=group, layout=layout, panel=panel_index, ABL=abl, abs_ILD=ild)
            audits.append(dict(**meta, display_trials=z['n'], parent_trials=z['parent_n'],
                signed_conditions=len(z['conditions']), model_window_mass=z['model_window_mass'], model_grid_area=area))
            histogram_rows.extend(dict(**meta, bin=i, left=z['edges'][i], right=z['edges'][i+1], count=int(n), density=d)
                for i,(n,d) in enumerate(zip(z['count'],z['data_density'])))
            curve_rows.extend(dict(**meta, rt_s=t, model_density=d, window_cdf=f)
                for t,d,f in zip(z['time'],z['model_density'],z['model_cdf']))
            model_bin_rows.extend(dict(**meta, bin=i, left=z['edges'][i], right=z['edges'][i+1], mass=m)
                for i,m in enumerate(z['model_bin_mass']))
            total += z['n']
        assert total==EXPECTED[group]['valid']+EXPECTED[group]['aborts']


# %% Six independent single-page PDFs; PNG previews live in a subfolder
ymax = float(np.ceil(max(max(z['model_density'].max(), z['data_density'].max()) for z in panels.values())*1.08))
artifacts = {}
for group in EXPECTED:
    for layout, rows, columns, selections, size in LAYOUTS:
        fig, axes = plt.subplots(rows, columns, figsize=size, sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=.065, right=.985, bottom=.23 if rows==1 else .11,
                            top=.77 if rows==1 else .88, wspace=.24, hspace=.38)
        for index, ((abl,ild), ax) in enumerate(zip(selections,axes.flat)):
            z = panels[group,layout,index]
            color = ABL_COLORS[abl] if abl is not None else ABL_COLORS[20]
            ax.stairs(z['data_density'], z['edges'], color=color, alpha=DATA_ALPHA, linewidth=DATA_WIDTH,
                      baseline=None, fill=False, zorder=2)
            ax.plot(z['time'], z['model_density'], color=MODEL_COLOR, linewidth=MODEL_WIDTH, zorder=3)
            ax.axvline(0,color='.75',linewidth=.6,zorder=0)
            ax.set(xlim=XLIM, ylim=(0,ymax), xticks=[-.1,0,.2,.4,.6])
            ax.set_title(f'|ILD| = {ild} dB' if ild is not None else f'ABL = {abl} dB')
            if index//columns==rows-1:ax.set_xlabel('Stimulus-relative RT (s)')
            if index%columns==0:
                ax.set_ylabel((f'ABL {abl} dB\n' if rows>1 else '')+'RT density (s⁻¹)')
            ax.text(.96,.94,f'n = {z["n"]:,}',transform=ax.transAxes,ha='right',va='top',fontsize=8.5)
            ax.grid(axis='y',alpha=.12)
        description = {'3x5_ABL_by_abs_ILD':'ABL × |ILD|',
            '1x5_ABL_pooled_by_abs_ILD':'ABLs pooled; separate |ILD|',
            '1x3_abs_ILD_pooled_by_ABL':'|ILD|s pooled; separate ABL'}[layout]
        e=EXPECTED[group]
        fig.suptitle(f'SHANK3 cohort 1 {group.upper()} | {description}\n'
            f'{e["animals"]} animals; {e["valid"]:,} valid choices + {e["aborts"]:,} pre-onset fixation aborts',
            fontsize=13,y=.98)
        fig.legend(handles=[Line2D([],[],color='.35',alpha=DATA_ALPHA,linewidth=DATA_WIDTH,label='Data: colored steps'),
                            Line2D([],[],color=MODEL_COLOR,linewidth=MODEL_WIDTH,label='Model: solid black')],
                   loc='upper center',bbox_to_anchor=(.5,.89 if rows==1 else .93),ncol=2,frameon=False)
        fig.text(.5,.025,'Gamma shared across ABLs; Omega per ABL; one reactive delay. Saved WT/HOM fits; no refitting.\n'
            '20 ms data bins; 1 ms model evaluation. Full [−1,1] s normalization; display −0.1 to 0.6 s.\n'
            'Pool raw data counts and parent-trial-weighted unnormalized model predictions before normalization.',
            ha='center',fontsize=8)
        stem=f'SHANK3_cohort1_{group.upper()}_RTD_{layout}'
        for path in [DEST/f'{stem}.pdf', DEST/'previews'/f'{stem}.png']:
            fig.savefig(path,dpi=160)
            artifacts[str(path.relative_to(DEST))]=sha256(path)
        plt.close(fig)
        info=subprocess.check_output(['pdfinfo',str(DEST/f'{stem}.pdf')],text=True)
        assert int(next(x.split(':')[1] for x in info.splitlines() if x.startswith('Pages:')))==1
        print(stem+'.pdf',flush=True)

# %% Numerical/count audits and preservation record
pd.DataFrame(audits).to_csv(DEST/'panel_audit.csv',index=False,float_format='%.17g')
pd.DataFrame(histogram_rows).to_csv(DEST/'data_histograms.csv',index=False,float_format='%.17g')
pd.DataFrame(curve_rows).to_csv(DEST/'model_curves.csv.gz',index=False,compression='gzip',float_format='%.17g')
pd.DataFrame(model_bin_rows).to_csv(DEST/'model_bin_masses.csv',index=False,float_format='%.17g')
for name,value in hashes.items():assert sha256(ROOT/name)==value,name
report=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),PDF_files=6,panels=46,
    display_trials_per_layout={'WT':81934,'HOM':80989},bin_width_s=.02,model_step_s=.001,
    normalization_window_s=[-1,1],display_window_s=list(XLIM),model_pooling='Exact parent frequencies; normalize after pooling',
    data_pooling='Sum counts before normalization',raw_trial_histograms_independently_verified=True,
    model_pooling_independently_verified=True,source_hashes_unchanged=True,source_hashes=hashes,artifacts=artifacts,
    visual_inspection='pending')
(DEST/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(DEST/'README.txt').write_text('Six RTD PDFs from the saved SHANK3 cohort 1 WT/HOM fits.\n'
    'Gamma shared across ABLs, Omega per ABL, one reactive delay. No refitting.\n'
    'For each genotype: 3x5 ABL x |ILD|; 1x5 ABL-pooled; 1x3 |ILD|-pooled.\n'
    'Valid choices plus pre-onset fixation aborts. Normalize over [-1,1] s; display [-0.1,0.6] s.\n'
    'Pool counts/model predictions with observed trial weights before normalization.\n'
    'PNG previews are in previews/; CSVs and validation.json document counts and pooling.\n'
    'Source: asd_analysis/export_shank3_gamma_by_ild_rtd_pdfs.py\n')
print('Saved all six PDFs to',DEST,flush=True)
