# %% Saved-fit sources, posterior convention and editable figure settings
"""Direct Gamma/Omega posterior dots versus the exact continuous NPL+alpha map.

The continuous curves are f(posterior mean parameters), not posterior means of
transformed draws. Gamma is ABL-invariant, so draw its common curve once.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64',True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fit_animal_by_animal'))
from numpyro_npl_alpha_svi_utils import gamma_omega_alpha_jax
SOURCES={
    'condition_gamma_omega':ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl',
    'npl_alpha':ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl'}
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_model_comparison'
ABLS=[20,40,60]
ILDS=[-16,-8,-4,-2,-1,1,2,4,8,16]
ILD_GRID=np.linspace(-16,16,321)
COLORS={20:'#1f77b4',40:'#ff7f0e',60:'#2ca02c'}
OFFSETS={20:-.18,40:0.,60:.18}  # Display jitter only; all calculations use true ILDs.
NPL_PARAMETERS=['rate_lambda','T_0','theta_E','rate_norm_l','alpha']
FIGSIZE,SIDE,DPI=(10.,10.2),3.4,200
STEM='SHANK3_cohort1_WT_HOM_Gamma_Omega_direct_vs_NPL_alpha_2x2'
OUT.mkdir(exist_ok=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


# %% Posterior dots and continuous derived curves; no refitting or CI smoothing
points,curves,parameter_rows=[],[],[]
hashes={};runs=[]
for group in ['wt','hom']:
    for model,folder in SOURCES.items():
        print('Reading',group,model,folder,flush=True)
        base=folder/group
        for name in ['fit_trials.csv.gz','conditions.csv','fixed_proactive.json','seed_0/configuration.json',
                     'seed_0/run_summary.json','seed_0/posterior_samples.npz','seed_0/posterior_summary.csv']:
            hashes[str((base/name).relative_to(ROOT))]=sha256(base/name)
        run=json.loads((base/'seed_0/run_summary.json').read_text())
        assert run['status']=='converged' and run['parameter_count']==(68 if model=='condition_gamma_omega' else 13)
        runs.append(dict(model=model,group=group,free_parameters=run['parameter_count'],restored_best_step=run['restored_best_step']))
    direct=SOURCES['condition_gamma_omega']/group
    npl=SOURCES['npl_alpha']/group
    conditions=pd.read_csv(direct/'conditions.csv')
    assert list(zip(conditions.ABL,conditions.ILD))==[(a,i) for a in ABLS for i in ILDS]
    saved_direct=pd.read_csv(SOURCES['condition_gamma_omega']/'parameter_estimates.csv',float_precision='round_trip')
    with np.load(direct/'seed_0/posterior_samples.npz') as samples:
        for block in ['gamma','omega']:
            x=samples[block]
            assert x.shape==(10000,30)
            for i,row in enumerate(conditions.itertuples()):
                mean=x[:,i].mean();lo,hi=np.quantile(x[:,i],[.025,.975])
                old=saved_direct.loc[saved_direct.group.eq(group)&saved_direct.block.eq(block)&saved_direct.ABL.eq(row.ABL)&saved_direct.ILD.eq(row.ILD)].iloc[0]
                np.testing.assert_allclose([mean,lo,hi],[old['mean'],old.lower95,old.upper95],atol=1e-12)
                points.append(dict(group=group,block=block,ABL=row.ABL,ILD=row.ILD,condition=i,
                    mean=mean,lower95=lo,upper95=hi,draws=10000,
                    displayed_ILD=row.ILD+OFFSETS[row.ABL],condition_trials=row.trials))
    with np.load(npl/'seed_0/posterior_samples.npz') as samples:
        means={name:float(samples[name].mean()) for name in NPL_PARAMETERS}
    for name,value in means.items(): parameter_rows.append(dict(group=group,parameter=name,posterior_mean=value))
    gamma,omega=[np.asarray(x) for x in gamma_omega_alpha_jax(np.asarray(ABLS)[:,None],ILD_GRID[None,:],**means)]
    np.testing.assert_allclose(gamma,gamma[0][None,:]+np.zeros((3,321)),atol=1e-13)
    for ai,abl in enumerate(ABLS):
        for i,ild in enumerate(ILD_GRID):
            curves.append(dict(group=group,ABL=abl,ILD=ild,gamma=gamma[ai,i],omega=omega[ai,i]))
points=pd.DataFrame(points);curves=pd.DataFrame(curves)
points.to_csv(OUT/'direct_gamma_omega_posterior_points.csv',index=False,float_format='%.17g')
curves.to_csv(OUT/'npl_alpha_continuous_gamma_omega.csv',index=False,float_format='%.17g')
pd.DataFrame(parameter_rows).to_csv(OUT/'npl_alpha_mapping_parameter_means.csv',index=False,float_format='%.17g')

# %% Equal square panels: genotype rows, Gamma/Omega columns
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42})
fig=plt.figure(figsize=FIGSIZE)
geometry,artists=[],[]
for row,group in enumerate(['wt','hom']):
    for col,block in enumerate(['gamma','omega']):
        panel=f'{group}_{block}'
        ax=fig.add_axes([(1.+4.6*col)/FIGSIZE[0],(5.35-4.1*row)/FIGSIZE[1],SIDE/FIGSIZE[0],SIDE/FIGSIZE[1]])
        # Same parameter scale across WT/HOM; retain all valid intervals/curves.
        bounds=points.loc[points.block.eq(block),['lower95','upper95']].to_numpy().ravel()
        bounds=np.r_[bounds,curves[block].to_numpy()]
        pad=max(np.ptp(bounds)*.10,.05)
        for ai,abl in enumerate(ABLS):
            z=curves.loc[curves.group.eq(group)&curves.ABL.eq(abl)].sort_values('ILD')
            if block=='omega' or ai==0:
                color=COLORS[abl] if block=='omega' else '#000000'
                ax.plot(z.ILD,z[block],color=color,linewidth=1.2,zorder=2)
                artists.append(dict(panel=panel,kind='curve',ABL=abl if block=='omega' else 'all',
                    color=color,ILD=z.ILD.tolist(),values=z[block].tolist()))
            z=points.loc[points.group.eq(group)&points.block.eq(block)&points.ABL.eq(abl)].sort_values('ILD')
            m,l,h=[z[k].to_numpy() for k in ['mean','lower95','upper95']]
            plotted=ax.errorbar(z.displayed_ILD,m,yerr=np.vstack([m-l,h-m]),fmt='o',color=COLORS[abl],
                markersize=4,capsize=2.5,elinewidth=.9,linestyle='none',zorder=3)
            np.testing.assert_allclose(np.asarray(plotted.lines[2][0].get_segments())[:,:,1],np.c_[l,h],atol=1e-12)
            artists.append(dict(panel=panel,kind='points',ABL=abl,color=COLORS[abl],ILD=z.ILD.tolist(),
                displayed_ILD=z.displayed_ILD.tolist(),mean=m.tolist(),lower95=l.tolist(),upper95=h.tolist()))
        ax.set(title=f'{group.upper()} | '+('Gamma' if block=='gamma' else 'Omega'),xlim=(-18,18),
            xticks=[-16,-8,-4,0,4,8,16],xlabel='Signed ILD (dB)',
            ylabel='Gamma (dimensionless)' if block=='gamma' else 'Omega (s⁻¹)',ylim=(min(bounds)-pad,max(bounds)+pad))
        size=ax.get_position().size*np.asarray(FIGSIZE)
        np.testing.assert_allclose(size,[SIDE,SIDE],atol=1e-12)
        geometry.append(dict(panel=panel,width_inches=float(size[0]),height_inches=float(size[1]),ylim=list(ax.get_ylim())))
fig.suptitle('SHANK3 cohort 1 | Direct Gamma/Omega versus NPL+alpha',y=.98,fontsize=15)
fig.legend([Line2D([0],[0],color=COLORS[a],marker='o',linestyle='none') for a in ABLS],
    [f'ABL {a} dB' for a in ABLS],ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.949))
fig.text(.5,.895,'Dots + 95% credible intervals: condition-specific fit  |  Lines: continuous NPL+alpha mapping\n'
    'Both fits use Gaussian sensory delays shared across ILDs within each ABL.',ha='center',fontsize=9)
fig.text(.5,.025,'NPL+alpha curves evaluate the exact mapping at posterior-mean parameters, every 0.1 dB.\n'
    'Its Gamma is ABL-invariant (one black curve); Omega curves use ABL colors. No curve uncertainty bands.\n'
    'Dots have ±0.18 dB display offsets only. Proactive/exponential lapses fixed; no refitting.',ha='center',fontsize=8.5)
for suffix in ['png','pdf']: fig.savefig(OUT/f'{STEM}.{suffix}',dpi=DPI)
plt.close(fig)
manifest=dict(status='generated_pending_verification',source_roots={k:str(v.relative_to(ROOT)) for k,v in SOURCES.items()},
    source_runs=runs,source_hashes=hashes,formula_source='fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py:gamma_omega_alpha_jax',
    formula_sha256=sha256(ROOT/'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py'),
    source_sha256=sha256(Path(__file__)),panels=4,direct_points=120,continuous_grid_points_per_ABL=321,
    curve_rows=len(curves),plotted_curves=8,curve_convention='exact mapping at physical posterior means, not transformed-draw average',
    Gamma_ABL_invariant=True,colors=COLORS,offsets_dB=OFFSETS,geometry=geometry,artists=artists,
    artifacts={f'{STEM}.{s}':sha256(OUT/f'{STEM}.{s}') for s in ['png','pdf']})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved 120 posterior dots and eight displayed continuous curves:',OUT,flush=True)
