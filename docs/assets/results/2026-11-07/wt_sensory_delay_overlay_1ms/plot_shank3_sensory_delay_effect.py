# %% Editable settings: WT NPL+alpha Gaussian-delay fit; no refitting
"""Model-only 1 ms overlays: sensory delay, zero-delay race, full fitted race.

The zero-delay curve bypasses the sensory-delay distribution only. Proactive
timing, exponential lapses and the NPL+alpha evidence/readout parameters stay
fixed. RT curves use the existing parent-onset averaging and [-1,1] window.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shank3_npl_alpha_gaussian_utils import ROOT, OUT, COUNTS, expanded, sha256
from shank3_wt_gamma_omega_utils import evidence_bound, ABLS, ILDS

GENOTYPE = 'wt'
N = COUNTS[GENOTYPE]
FIT = OUT/GENOTYPE
BASE = FIT/'sensory_delay_effect'
DEST = BASE/'overlay_1ms'
BIN_WIDTH = .020
RT_GRID = np.linspace(-1., 1., 2001)
RT_EDGES = np.linspace(-1., 1., 101)
DELAY_GRID = np.linspace(0., 1., 1001)
DELAY_EDGES = np.linspace(0., 1., 51)
RT_XLIM = [0., .4]
COLORS = {'sensory_delay':'#1f77b4','zero_sensory_delay_rt':'#ff7f0e','original_rt':'#2ca02c'}
FIGSIZE, DPI = (32., 11.), 180
STEM = f'SHANK3_cohort1_{GENOTYPE.upper()}_sensory_delay_model_overlay_3x10_1ms'
DEST.mkdir(parents=True, exist_ok=True)

# %% Preserve sources and read the existing physical posterior means
inputs = [FIT/name for name in ['seed_0/posterior_samples.npz', 'conditions.csv',
    'fixed_proactive.json', 'fixed_parent_design_averages.npz',
    'fixed_parent_design_manifest.json', 'signed_parent_model_1ms.csv.gz',
    'condition_parameter_means.csv', 'parent_trials.csv.gz', 'fit_trials.csv.gz']]
sources = {str(p.relative_to(ROOT)):sha256(p) for p in inputs}
samples = dict(np.load(FIT/'seed_0/posterior_samples.npz'))
params = {name:value.mean(axis=0) for name,value in samples.items()}
mapped = {name:np.asarray(value) for name,value in expanded(params).items()}
conditions = pd.read_csv(FIT/'conditions.csv', float_precision='round_trip')
original = pd.read_csv(FIT/'signed_parent_model_1ms.csv.gz', float_precision='round_trip')
reference_params = pd.read_csv(FIT/'condition_parameter_means.csv', float_precision='round_trip')
fixed = json.loads((FIT/'fixed_proactive.json').read_text())['values']
v, bound, proactive_delay, lapse_prob, beta = fixed
cache = dict(np.load(FIT/'fixed_parent_design_averages.npz'))
np.testing.assert_array_equal(cache.pop('grid'), RT_GRID)
cache_manifest = json.loads((FIT/'fixed_parent_design_manifest.json').read_text())
for name,digest in cache_manifest['sources'].items():
    assert sha256(FIT/name) == digest
assert conditions.parent_trials.sum() == N['parent'] and conditions.trials.sum() == N['valid']
for name in ['gamma','omega','delay_mu','delay_sigma']:
    np.testing.assert_allclose(mapped[name],reference_params[name],atol=1e-12,rtol=1e-12)

# %% Per signed ABL/ILD: original delay, exact zero-delay race and saved full race
histograms, curves, audits, parameter_rows = [], [], [], []
for row in conditions.itertuples():
    i = row.condition
    gamma, omega = mapped['gamma'][i], mapped['omega'][i]
    mu, sd = mapped['delay_mu'][i], mapped['delay_sigma'][i]
    # Sum the two COMPLETE choice densities: f_A*S_E + S_A*f_E plus lapse.
    left = [np.asarray(x) for x in evidence_bound(RT_GRID,gamma,omega,mapped['w'],-1)]
    right = [np.asarray(x) for x in evidence_bound(RT_GRID,gamma,omega,mapped['w'],1)]
    evidence_pdf = left[0]+right[0]
    evidence_survival = left[2]+right[2]
    ap, ass, lp, ls = [cache[f'{i}_{name}'] for name in
        ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']]
    zero_pdf = (1-lapse_prob)*(ap*evidence_survival+ass*evidence_pdf)+lapse_prob*lp
    zero_cdf = 1-(1-lapse_prob)*ass*evidence_survival-lapse_prob*ls
    old = original.loc[original.condition.eq(i)].sort_values('rt_s')
    np.testing.assert_array_equal(old.rt_s, RT_GRID)
    assert len(old) == 2001 and old.parent_trials.eq(row.parent_trials).all()
    delay = truncnorm(-mu/sd, np.inf, loc=mu, scale=sd)
    distributions = [
        ('sensory_delay',DELAY_GRID,delay.pdf(DELAY_GRID),delay.cdf(DELAY_GRID),DELAY_EDGES,1.),
        ('zero_sensory_delay_rt',RT_GRID,zero_pdf,zero_cdf,RT_EDGES,zero_cdf[-1]-zero_cdf[0]),
        ('original_rt',RT_GRID,old.unnormalized_density.to_numpy(),
         old.unconditional_cdf.to_numpy(),RT_EDGES,old.unconditional_cdf.iloc[-1]-old.unconditional_cdf.iloc[0])]
    for label,grid,pdf,cdf,edges,mass in distributions:
        assert mass > 0
        edge_cdf = cdf[::20]
        bins = np.diff(edge_cdf)/mass
        assert bins.min() > -1e-10
        for j,(lo,hi,prob) in enumerate(zip(edges[:-1],edges[1:],bins)):
            histograms.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,distribution=label,
                bin=j,left_s=lo,right_s=hi,probability=prob,density=prob/BIN_WIDTH))
        for t,raw_pdf,raw_cdf in zip(grid,pdf,cdf):
            curves.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,distribution=label,
                time_s=t,unnormalized_density=raw_pdf,unconditional_cdf=raw_cdf,
                density=raw_pdf/mass))
        audits.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,distribution=label,
            parent_trials=row.parent_trials,fit_valid_trials=row.trials,
            normalization_mass=mass,histogram_area=bins.sum(),
            mass_beyond_1s=delay.sf(1.) if label=='sensory_delay' else np.nan))
    parameter_rows.append(dict(condition=i,ABL=row.ABL,ILD=row.ILD,
        gamma_derived=gamma,omega_derived=omega,delay_mu_s=mu,delay_sigma_s=sd,
        zero_delay_s=0.,w=float(mapped['w']),del_go_s=float(mapped['del_go'])))

histograms = pd.DataFrame(histograms)
curves = pd.DataFrame(curves)
audits = pd.DataFrame(audits)
histograms.to_csv(DEST/'model_histogram_bins.csv',index=False)
curves.to_csv(DEST/'model_curves_1ms.csv.gz',index=False,compression='gzip')
audits.to_csv(DEST/'condition_audit.csv',index=False)
pd.DataFrame(parameter_rows).to_csv(DEST/'condition_parameters.csv',index=False)

# %% Three ABL rows, ten signed-ILD columns; three model densities per panel
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
fig, axes = plt.subplots(3,10,figsize=FIGSIZE,sharex=True,sharey=True)
fig.subplots_adjust(left=.035,right=.993,bottom=.12,top=.84,hspace=.28,wspace=.18)
labels = ['sensory_delay','zero_sensory_delay_rt','original_rt']
titles = ['Sensory delay / NDT','Full model, sensory delay = 0','Full model RT']
visible = curves.loc[curves.time_s.between(*RT_XLIM)]
ymax = visible.density.max()*1.10
geometry = []
artists = []
for r,abl in enumerate(ABLS):
    for c,ild in enumerate(ILDS):
        ax = axes[r,c]
        for label,title in zip(labels,titles):
            z = curves.loc[curves.ILD.eq(ild)&curves.ABL.eq(abl)&
                curves.distribution.eq(label)].sort_values('time_s')
            ax.plot(z.time_s,z.density,color=COLORS[label],linewidth=1.2,label=title)
            artists.append(dict(row=r,column=c,condition=int(z.condition.iloc[0]),ABL=abl,ILD=ild,
                distribution=label,color=COLORS[label],points=len(z),grid_step_s=.001,
                time_sha256=hashlib.sha256(z.time_s.to_numpy().tobytes()).hexdigest(),
                density_sha256=hashlib.sha256(z.density.to_numpy().tobytes()).hexdigest()))
        ax.set_xlim(RT_XLIM)
        ax.set_ylim(0,ymax)
        ax.set_box_aspect(1)
        ax.set_xticks([0,.1,.2,.3,.4])
        if r==0: ax.set_title(f'ILD {ild:+g} dB',fontsize=12,pad=12)
        if c==0: ax.set_ylabel(f'ABL {abl} dB\nDensity (s⁻¹)',fontsize=12)
        geometry.append(dict(row=r,column=c,ABL=abl,ILD=ild,
            xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
fig.suptitle(f'SHANK3 cohort 1 {GENOTYPE.upper()} | NPL+alpha | sensory-delay effect\n'
    f'{N["animals"]} animals; model-only densities at 1 ms resolution; no refitting',fontsize=17,y=.982)
fig.legend(handles=[Line2D([0],[0],color=COLORS[k],lw=1.8,label=t) for k,t in zip(labels,titles)],
    loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=13)
fig.supxlabel('Time (s): sensory delay for NDT; stimulus-relative RT for the two full-model curves',y=.075,fontsize=12)
fig.text(.5,.022,'All three curves are model predictions. Proactive timing, exponential lapses and NPL+alpha evidence/readout parameters remain fixed; only the orange curve removes sensory delay.\n'
    'Each RT curve uses exact parent-onset averages and its own [−1,1] s normalization. NDT is normalized over [0,∞), without RT-window conditioning. Display 0–0.4 s; signed ILDs remain separate.',
    ha='center',fontsize=10)
fig.savefig(DEST/f'{STEM}.png',dpi=DPI)
fig.savefig(DEST/f'{STEM}.pdf')
plt.close(fig)
for name,digest in sources.items():
    assert sha256(ROOT/name) == digest
manifest = dict(status='generated',genotype=GENOTYPE.upper(),model='NPL+alpha',animals=N['animals'],
    fitted_valid_trials=N['valid'],parent_trials=N['parent'],conditions=30,rows=3,columns=10,
    signed_ILDS=ILDS,ABLs=ABLS,colors=COLORS,model_grid_step_s=.001,
    display_kind='continuous model-density curves, not empirical histograms',
    reference_histogram_audit_bin_width_s=BIN_WIDTH,
    RT_normalization_window_s=[-1,1],RT_display_window_s=RT_XLIM,
    NDT_normalization='underlying zero-truncated Gaussian on [0,infinity), no RT conditioning',
    fixed_proactive_values=fixed,remaining_parameters_unchanged=True,
    calculation='complete f_right+f_left; Gaussian bypassed for zero-delay curve only',
    parent_weighting='every exact original parent onset; no signed-condition pooling',
    source_hashes=sources,script_sha256=sha256(Path(__file__)),geometry=geometry,artists=artists,
    artifacts={f'{STEM}.{ext}':sha256(DEST/f'{STEM}.{ext}') for ext in ['png','pdf']})
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Created {DEST/STEM}.png and .pdf; 3 ABL rows x 10 signed ILDs; 90 model curves at 1 ms.',flush=True)
