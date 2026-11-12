# %% ASD0047 joint report: all parameters fitted together, no frozen abort estimates
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import invgauss
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from shank3_asd0047_joint_utils import ROOT, OUT, BASE, OLD, ABLS, ILDS, sha256, evidence_grid, PARAMETER_COUNT, FIXED_PARAMETERS, VARIANT

# %% Editable appearance and numerical settings
COLORS = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c'}
ABS_ILDS = [1, 2, 4, 8, 16]
RT = np.linspace(-1, 1, 2001)
EDGES = np.linspace(-1, 1, 101)
XLIM = [-.1, .6]
DPI, CHUNK, QUADRATURE_ORDER = 180, 128, 24
PREFIX = 'SHANK3_cohort1_ASD0047_joint_NPL_alpha_Gaussian_ABL_' + VARIANT
REPORT = OUT / 'report'
DELIVERY = ROOT / 'shan3_coh_1_animal_Wise_fitting/ASD0047_joint'
REPORT.mkdir(exist_ok=True)
selected = json.loads((OUT / 'selected_fit.json').read_text())
assert selected['status'] in ['passed', 'exponential_weight_requires_nested_refit']
run = ROOT / selected['selected_run']
summary = json.loads((run / 'run_summary.json').read_text())
config = json.loads((run / 'configuration.json').read_text())
prep = json.loads((OUT / 'preparation.json').read_text())
samples = dict(np.load(run / 'posterior_samples.npz'))
p = {key: value.mean(0) for key, value in samples.items()}
p.update(FIXED_PARAMETERS)
delay_moments = []
for j, abl in enumerate(ABLS):
    mu, sd = samples['delay_mu'][:, j], samples['delay_sigma'][:, j]
    z = mu/sd
    ratio = np.exp(-.5*z*z)/np.sqrt(2*np.pi)/ndtr(z)
    actual_mean = mu+sd*ratio
    actual_sd = sd*np.sqrt(1-z*ratio-ratio**2)
    for label, values in [('truncated_mean', actual_mean), ('truncated_sd', actual_sd)]:
        delay_moments.append(dict(ABL=abl, moment=label, posterior_mean_s=values.mean(),
            q025_s=np.quantile(values, .025), q975_s=np.quantile(values, .975)))
pd.DataFrame(delay_moments).to_csv(REPORT / 'actual_truncated_delay_moments.csv', index=False, float_format='%.17g')
parent = pd.read_csv(OUT / 'parent_trials.csv.gz', float_precision='round_trip')
valid = pd.read_csv(OUT / 'fit_trials.csv.gz', float_precision='round_trip')
conditions = pd.read_csv(OUT / 'conditions.csv')
assert len(parent) == 21786 and len(valid) == 18771
source_hashes = {str(q.relative_to(ROOT)): sha256(q) for q in [run / 'posterior_samples.npz',
    OUT / 'parent_trials.csv.gz', OUT / 'fit_trials.csv.gz', OUT / 'observations.npz',
    OUT / 'likelihood_policy.json', OUT / 'selected_fit.json', Path(__file__)]}
dt = .0001 if np.min(p['delay_sigma']) < .0025 else .0005
eg, ef, ec = [np.asarray(x) for x in evidence_grid(p, dt=dt, grid_end=config['grid_end'])]
FP = [CubicHermiteSpline(eg, x, np.gradient(x, eg, axis=-1), axis=-1) for x in ef]
FC = [CubicHermiteSpline(eg, x, np.gradient(x, eg, axis=-1), axis=-1) for x in ec]
v, a, nd = [float(p[k]) for k in ['V_A', 'theta_A', 't_nd']]
lapse, q, d, beta = [float(p[k]) for k in ['lapse_prob', 'lapse_prob_right', 'lapse_exp_weight', 'beta_lapse']]
CUTOFF = prep['contaminant_cutoff_s']


def contaminant_pdf_sf(t):
    u = np.clip(t, 0, CUTOFF)
    z = -np.expm1(-beta*CUTOFF)
    pdf = np.where((t >= 0) & (t <= CUTOFF), d*beta*np.exp(-beta*u)/z+(1-d)/CUTOFF, 0.)
    sf = d*np.exp(-beta*u)*(-np.expm1(-beta*(CUTOFF-u)))/z+(1-d)*(1-u/CUTOFF)
    return pdf, sf


def evidence(times, condition):
    f = np.maximum(FP[condition](np.clip(times, 0, eg[-1])), 0.)
    c = np.clip(FC[condition](np.clip(times, 0, eg[-1])), 0., 1.)
    cg = np.clip(FC[condition](np.clip(times+float(p['del_go']), 0, eg[-1])), 0., 1.)
    f[:, times <= 0] = 0.
    c[:, times <= 0] = 0.
    cg[:, times+float(p['del_go']) <= 0] = 0.
    return f.sum(0), 1-c.sum(0), f[1], .5*(1-cg.sum(0))+cg[1]-c[1]


def averaged_components(times, onsets, weights):
    """Every exact onset; weights are inverse window mass, not renormalized."""
    result = np.empty((3, len(times)))
    for first in range(0, len(times), CHUNK):
        section = slice(first, first+CHUNK)
        absolute = times[section, None]+onsets[None, :]
        ap = invgauss.pdf(absolute-nd, mu=1/(v*a), scale=a*a)
        ass = invgauss.sf(absolute-nd, mu=1/(v*a), scale=a*a)
        ap = np.where(absolute >= 0, ap, 0.)
        ass = np.where(absolute >= 0, ass, 0.)
        h, _ = contaminant_pdf_sf(absolute)
        result[:, section] = np.array([(x*weights[None, :]).mean(1) for x in [ap, ass, h]])
    return result


# %% Exact onset-specific normalization, parent pooling and valid-window choices
signed, normalizers, probabilities, quadrature_checks = [], [], [], []
for row in conditions.itertuples():
    i = row.condition
    z = parent.loc[parent.condition.eq(i)]
    onset = z.onset_s.to_numpy()
    end_sf = evidence(np.array([1.]), i)[1][0]
    s0 = (1-lapse)*invgauss.sf(-nd, mu=1/(v*a), scale=a*a)+lapse
    send = (1-lapse)*invgauss.sf(onset+1-nd, mu=1/(v*a), scale=a*a)*end_sf+lapse*contaminant_pdf_sf(onset+1)[1]
    retained_mass = s0-send
    assert np.all((retained_mass > 0) & (retained_mass <= 1))
    ap, ass, h = averaged_components(RT, onset, 1/retained_mass)
    ep, es, _, _ = evidence(RT, i)
    density = (1-lapse)*(ap*es+ass*ep)+lapse*h
    for t, value in zip(RT, density):
        signed.append(dict(condition=i, ABL=row.ABL, ILD=row.ILD, rt_s=t,
            retained_parent_trials=len(z), density=value))
    normalizers.extend(dict(prepared_csv_line=line, condition=i, onset_s=c,
        entry_survival=s0, retained_window_mass=m) for line, c, m in zip(z.prepared_csv_line, onset, retained_mass))

    z = valid.loc[valid.condition.eq(i)]
    onset = z.onset_s.to_numpy()
    sc = (1-lapse)*invgauss.sf(onset-nd, mu=1/(v*a), scale=a*a)+lapse*contaminant_pdf_sf(onset)[1]
    se = (1-lapse)*invgauss.sf(onset+1-nd, mu=1/(v*a), scale=a*a)*end_sf+lapse*contaminant_pdf_sf(onset+1)[1]
    window = sc-se
    assert np.all(window > 0)
    integrals = []
    cuts = np.linspace(0, 1, 21)
    for order in [QUADRATURE_ORDER, 2*QUADRATURE_ORDER]:
        nodes, weights = leggauss(order)
        times = np.concatenate([.5*(hi-lo)*nodes+.5*(hi+lo) for lo, hi in zip(cuts[:-1], cuts[1:])])
        weights = np.concatenate([.5*(hi-lo)*weights for lo, hi in zip(cuts[:-1], cuts[1:])])
        ap, ass, h = averaged_components(times, onset, 1/window)
        ep, es, er, readout = evidence(times, i)
        right = (1-lapse)*(ap*readout+ass*er)+lapse*q*h
        total = (1-lapse)*(ap*es+ass*ep)+lapse*h
        integrals.append([float(weights@right), float(weights@total)])
    np.testing.assert_allclose(integrals[0], integrals[1], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(integrals[1][1], 1, atol=2e-6)
    probabilities.append(dict(condition=i, ABL=row.ABL, ILD=row.ILD, trials=len(z),
        right=int(z.response_poke.eq(1).sum()), data_p_right=z.response_poke.eq(1).mean(),
        model_p_right=integrals[-1][0]))
    quadrature_checks.append(dict(condition=i, low_order=integrals[0], high_order=integrals[1]))
    print(f'Condition {i+1}/30: retained parents={row.parent_trials}, model P(right)={integrals[-1][0]:.6f}', flush=True)
signed = pd.DataFrame(signed)
probabilities = pd.DataFrame(probabilities)
signed.to_csv(REPORT / 'signed_model_densities.csv.gz', index=False, float_format='%.17g')
pd.DataFrame(normalizers).to_csv(REPORT / 'onset_window_normalizers.csv.gz', index=False, float_format='%.17g')
probabilities.to_csv(REPORT / 'psychometric_probabilities.csv', index=False, float_format='%.17g')

# %% Pool actual signed trials before separately normalizing data and model panels
display = parent.loc[parent.fit_valid | parent.display_abort].copy()
display['rt'] = np.where(display.fit_valid, display.timed_rt, display.abort_rt_s)
assert len(display) == 21694 and int(display.display_abort.sum()) == 2923
histograms, curves, masses = [], [], []
for abl in ABLS:
    for magnitude in ABS_ILDS:
        z = display.loc[display.ABL_mapped.eq(abl) & display.ILD.abs().eq(magnitude)]
        counts, _ = np.histogram(z.rt, EDGES)
        assert counts.sum() == len(z)
        for j, count in enumerate(counts):
            histograms.append(dict(ABL=abl, abs_ILD=magnitude, bin=j, left=EDGES[j], right=EDGES[j+1],
                count=count, density=count/len(z)/.020, display_trials=len(z)))
        block = signed.loc[signed.ABL.eq(abl) & signed.ILD.abs().eq(magnitude)]
        weighted = block.assign(weighted=block.density*block.retained_parent_trials)
        pooled = weighted.groupby('rt_s', sort=True).weighted.sum().to_numpy()/block.groupby('ILD').retained_parent_trials.first().sum()
        mass = float(simpson(pooled, x=RT))
        assert mass > 0
        normalized = pooled/mass
        np.testing.assert_allclose(simpson(normalized, x=RT), 1, atol=1e-12)
        masses.append(dict(ABL=abl, abs_ILD=magnitude, model_pre_normalization_mass=mass,
            data_window_count=len(z), retained_parent_count=int(block.groupby('ILD').retained_parent_trials.first().sum())))
        curves.extend(dict(ABL=abl, abs_ILD=magnitude, rt_s=t, density=f) for t, f in zip(RT, normalized))
histograms, curves = pd.DataFrame(histograms), pd.DataFrame(curves)
histograms.to_csv(REPORT / 'rtd_data_bins.csv', index=False, float_format='%.17g')
curves.to_csv(REPORT / 'rtd_model_curves.csv.gz', index=False, float_format='%.17g')
pd.DataFrame(masses).to_csv(REPORT / 'rtd_window_masses.csv', index=False, float_format='%.17g')

# %% Two-page PDF and matching PNGs; x zoom never changes normalization
plt.rcParams.update({'font.size': 11, 'axes.grid': False, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharex=True, sharey=True)
visible = curves.loc[curves.rt_s.between(*XLIM)]
YMAX = 1.08*max(float(histograms.density.max()), float(visible.density.max()))
for r, abl in enumerate(ABLS):
    for c, magnitude in enumerate(ABS_ILDS):
        ax = axes[r, c]
        z = histograms.loc[histograms.ABL.eq(abl) & histograms.abs_ILD.eq(magnitude)]
        m = curves.loc[curves.ABL.eq(abl) & curves.abs_ILD.eq(magnitude)]
        ax.plot(m.rt_s, m.density, color='black', lw=.9, zorder=2)
        ax.stairs(z.density, EDGES, color=COLORS[abl], lw=1.4, zorder=3)
        ax.set(xlim=XLIM, ylim=(0, YMAX), title=f'ABL {abl} dB; |ILD| {magnitude} dB\nn = {int(z.display_trials.iloc[0]):,}')
        if r == 2:
            ax.set_xlabel('RT from stimulus onset (s)')
        if c == 0:
            ax.set_ylabel('Density (s⁻¹)')
fig.suptitle('SHANK3 cohort 1 WT ASD0047 — simultaneous proactive + NPL+α + contaminants\nASD0047; 18,771 valid responses + 2,923 diagnostic pre-onset aborts', y=.99)
fig.legend([Line2D([], [], color=COLORS[20], lw=1.4), Line2D([], [], color='black', lw=.9)],
    ['Data (ABL color; 20 ms bins)', 'Complete fitted model (1 ms)'], loc='lower center', bbox_to_anchor=(.5, .072), ncol=2)
boundary_note = (' Physical-domain review: '+', '.join(selected['intrinsic_flags'])+'.') if selected['intrinsic_flags'] else ''
caption = (f'All {PARAMETER_COUNT} free parameters fitted jointly; 161 valid RTs >1 s excluded. Likelihood conditions on 0 ≤ fixation-time response ≤ planned onset+1 s.\n'
    'Each model prediction is normalized per exact onset, then pooled by retained parent trials (21,786). Late fixation-labelled rows are onset-censored.\n'
    'RTD data/model are separately normalized over [−1,1] s; display only −0.1–0.6 s. No repeat filter is reapplied.'+boundary_note)
fig.text(.5, .012, caption, ha='center', va='bottom', fontsize=9)
fig.tight_layout(rect=[0, .15, 1, .98])
fig.savefig(REPORT / (PREFIX+'_RTDs.png'), dpi=DPI)

psy, ax = plt.subplots(figsize=(10, 8))
for abl in ABLS:
    z = probabilities.loc[probabilities.ABL.eq(abl)].sort_values('ILD')
    ax.plot(z.ILD, z.model_p_right, color=COLORS[abl], lw=1.2)
    ax.scatter(z.ILD, z.data_p_right, color=COLORS[abl], s=35, zorder=3, label=f'ABL {abl} dB (n={z.trials.sum():,})')
ax.set(xlim=(-18, 18), ylim=(0, 1), xlabel='Signed ILD (dB)', ylabel='P(right)',
    title='SHANK3 cohort 1 WT ASD0047 — joint-fit psychometric\n18,771 fitted valid choices; no additional sigmoid')
ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
ax.legend(loc='upper left')
psy.text(.5, .02, 'Dots: observed choices. Lines: onset-specific complete-mixture P(right | valid RT ∈ [0,1] s), averaged over fitted trials.\n'
    'Posterior-mean predictions; proactive, reactive and contaminants estimated together. Lines join measured stimulus conditions.',
    ha='center', fontsize=9)
psy.tight_layout(rect=[0, .09, 1, 1])
psy.savefig(REPORT / (PREFIX+'_psychometric.png'), dpi=DPI)
pdf_path = REPORT / (PREFIX+'_report.pdf')
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
    pdf.savefig(psy)
plt.close(fig)
plt.close(psy)

manifest = dict(status='generated_pending_independent_and_visual_checks', selected_run=str(run.relative_to(ROOT)),
    pdf=str(pdf_path), pngs=[PREFIX+'_RTDs.png', PREFIX+'_psychometric.png'], pages=2,
    fitted_parameters=PARAMETER_COUNT, valid_choices=18771, aborts_in_likelihood=3004,
    censored=11, excluded_above_1=161, retained_parent_design=21786, displayed=21694,
    rtd_panels=15, psychometric_points=30, bins_s=.02, model_step_s=.001,
    calculation_window_s=[-1, 1], display_xlim_s=XLIM, density_ylim=[0, YMAX], quadrature_checks=quadrature_checks,
    bound_acceptance=selected, source_hashes=source_hashes,
    delivery_folder=str(DELIVERY))
(REPORT / 'report_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(str(pdf_path), flush=True)
