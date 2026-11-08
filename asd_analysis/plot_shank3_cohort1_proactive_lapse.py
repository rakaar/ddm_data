# %% Imports and editable plotting parameters
"""Diagnostics for pooled no-LED proactive/lapse fits, with exact censor exposure.

The empirical denominator is ALL eligible trials, not observed aborts. Theory
uses the selected start's posterior mean; the parameter figure shows marginal
central 95% variational-posterior intervals, not SEM across animals.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import invgauss

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'asd_analysis/shank3_cohort1_proactive_lapse'
GENOTYPES = ['wt', 'hom', 'het']
COLORS = {'wt':'tab:blue', 'hom':'tab:green', 'het':'tab:orange'}
PARAMETERS = ['V_A', 'theta_A', 't_nd', 'lapse_prob', 'beta_lapse']
PARAM_LABELS = ['$V_A$ (drift; diffusion = 1)', '$\\theta_A$ (bound; diffusion = 1)',
                '$t_{nd}$ (s)', 'Lapse probability', '$\\beta_{lapse}$ (s⁻¹)']
BIN_WIDTH = .020
MODEL_GRID_STEP = .001
PNG_DPI = 240
STEMS = ['SHANK3_cohort1_proactive_lapse_abort_distributions',
         'SHANK3_cohort1_proactive_lapse_convergence',
         'SHANK3_cohort1_proactive_lapse_parameters']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
prep = json.loads((OUT/'preparation.json').read_text())
selected = json.loads((OUT/'selected_starts.json').read_text())
trials = pd.read_csv(OUT/'trial_classifications.csv.gz', float_precision='round_trip')
selection = {r['genotype']:r for r in selected}
parameter_tables = {}
for g in GENOTYPES:
    parameter_tables[g] = pd.read_csv(OUT/g/f'seed_{selection[g]["seed"]}/posterior_summary.csv').set_index('parameter')


def components(t, params):
    """Weighted proactive/lapse PDF and CDF, computed independently with SciPy."""
    v, a, delay, p, beta = params
    t = np.asarray(t)
    active = t > delay
    apdf, acdf = np.zeros_like(t), np.zeros_like(t)
    apdf[active] = (1-p)*invgauss.pdf(t[active]-delay, mu=1/(v*a), scale=a*a)
    acdf[active] = (1-p)*invgauss.cdf(t[active]-delay, mu=1/(v*a), scale=a*a)
    return apdf, p*beta*np.exp(-beta*t), acdf, p*(-np.expm1(-beta*t))


# %% Counts/(all eligible trials * width), and exact trialwise-censoring model mass
last_edge = np.ceil(trials.censor_s.max()/BIN_WIDTH)*BIN_WIDTH
edges = np.arange(round(last_edge/BIN_WIDTH)+1)*BIN_WIDTH
grid = np.linspace(0, last_edge, round(last_edge/MODEL_GRID_STEP)+1)
histograms, model_tables, probabilities = [], [], []
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.72, wspace=.15)
for ax, g in zip(axes, GENOTYPES):
    z = trials.loc[trials.genotype.eq(g)]
    n, event_n = len(z), int(z.is_event.sum())
    p = parameter_tables[g].loc[PARAMETERS, 'mean'].to_numpy()
    c = np.sort(z.censor_s.to_numpy())
    apdf, lpdf, _, _ = components(grid, p)
    at_risk_fraction = (n-np.searchsorted(c, grid, side='right'))/n
    proactive, lapse = apdf*at_risk_fraction, lpdf*at_risk_fraction
    # G(t) = mean F(min(t,c_i)); evaluate exactly via sorted c and prefix sums.
    _, _, a_at_c, l_at_c = components(c, p)
    prefix = np.r_[0., np.cumsum(a_at_c+l_at_c)]
    _, _, a_edges, l_edges = components(edges, p)
    ended = np.searchsorted(c, edges, side='right')
    cumulative = (prefix[ended]+(n-ended)*(a_edges+l_edges))/n
    bin_mass = np.diff(cumulative)
    event_counts = np.histogram(z.loc[z.is_event, 'fixation_s'], edges)[0]
    empirical = event_counts/(n*BIN_WIDTH)
    assert event_counts.sum() == event_n and np.all(bin_mass >= -1e-14)
    observed, predicted = event_n/n, float(np.mean(a_at_c+l_at_c))
    np.testing.assert_allclose(bin_mass.sum(), predicted, atol=1e-12)
    ax.stairs(empirical, edges, color=COLORS[g], linewidth=1.8, label='Data (20 ms bins)')
    ax.plot(grid, proactive+lapse, color='black', lw=1.9, label='Mixture')
    ax.plot(grid, proactive, color='dimgray', ls='--', lw=1.7, label='Proactive component')
    ax.plot(grid, lapse, color='tab:purple', ls=':', lw=1.9, label='Exponential-lapse component')
    ax.set(title=f'{g.upper()} | {z.animal.nunique()} animals | {n:,} trials\n'
        f'Pre-onset aborts: data {observed:.2%}; model {predicted:.2%}',
        xlabel='Time from fixation onset (s)', xlim=(0, last_edge), ylim=(0, None))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    histograms.append(pd.DataFrame({'genotype':g, 'bin_left_s':edges[:-1], 'bin_right_s':edges[1:],
        'event_count':event_counts, 'all_eligible_trials':n, 'empirical_density':empirical,
        'model_bin_mass':bin_mass, 'model_bin_average_density':bin_mass/BIN_WIDTH}))
    model_tables.append(pd.DataFrame({'genotype':g, 'time_s':grid, 'at_risk_fraction':at_risk_fraction,
        'proactive_density':proactive, 'lapse_density':lapse, 'mixture_density':proactive+lapse}))
    probabilities.append({'genotype':g, 'animals':z.animal.nunique(), 'eligible_trials':n,
        'events':event_n, 'observed_abort_probability':observed, 'predicted_abort_probability':predicted,
        'predicted_proactive_abort_probability':float(a_at_c.mean()), 'predicted_lapse_abort_probability':float(l_at_c.mean()),
        'selected_seed':selection[g]['seed'], 'fit_status':selection[g]['status']})
axes[0].set_ylabel('Pre-onset abort density per eligible trial (s⁻¹)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.86), ncol=4, frameon=False, fontsize=10)
fig.suptitle('SHANK3 cohort 1 | Proactive + exponential-lapse fits', y=.98, fontsize=17)
fig.text(.5,.065,'Standard RT session type 1; training 16; ABLs and signed ILDs pooled. No measured-RT restriction.\n'
    'Histogram area = observed pre-onset abort fraction. Model components include each trial’s exact planned-onset censoring.\n'
    'Valid choices and fixation-labelled trials at/after onset are censored. Curves use selected-start posterior means.',
    ha='center', fontsize=9.4, linespacing=1.5)
for extension in ['png','pdf']:
    fig.savefig(OUT/(STEMS[0]+'.'+extension), dpi=PNG_DPI, facecolor='white')
plt.close(fig)
pd.concat(histograms).to_csv(OUT/'diagnostic_histogram_bins.csv', index=False)
pd.concat(model_tables).to_csv(OUT/'diagnostic_model_curves.csv', index=False)
pd.DataFrame(probabilities).to_csv(OUT/'abort_probability_comparison.csv', index=False)

# %% Both starts, absolute-best restored window, and final checked step
fig, axes = plt.subplots(1,3, figsize=(17,5.9))
fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.72, wspace=.29)
for ax, g in zip(axes, GENOTYPES):
    lines = []
    traces = []
    for seed, color in [(0,'tab:blue'),(1,'tab:orange')]:
        directory = OUT/g/f'seed_{seed}'
        s = json.loads((directory/'run_summary.json').read_text())
        windows = pd.read_csv(directory/'convergence_windows.csv')
        ax.plot(windows.end_step/1000, windows.mean_negative_elbo, color=color, lw=1.2, label=f'Seed {seed}')
        ax.scatter(s['restored_best_step']/1000, s['restored_window_negative_elbo'], color=color, marker='*', s=135, zorder=5)
        ax.scatter(s['final_checked_step']/1000, windows.iloc[-1].mean_negative_elbo, color=color, marker='x', s=50, zorder=5)
        lines.append(f'Seed {seed}: restored {s["restored_best_step"]:,}; checked {s["final_checked_step"]:,} ({s["status"]})')
        traces.append((windows, s, color))
    ax.set(title=f'{g.upper()} | selected seed {selection[g]["seed"]}', xlabel='SVI steps (thousands)', ylabel='Mean window negative ELBO')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16)
    ax.text(.5,-.23,'\n'.join(lines), ha='center',va='top',transform=ax.transAxes,fontsize=8.3,linespacing=1.6)
    # The displaced start's initial loss hides the plateau on the full-range axis.
    inset = ax.inset_axes([.25,.42,.7,.5])
    baseline = min(s['restored_window_negative_elbo'] for _,s,_ in traces)
    for windows,s,color in traces:
        tail = windows.loc[windows.end_step >= 20000]
        inset.plot(tail.end_step/1000,tail.mean_negative_elbo-baseline,color=color,lw=1)
        inset.scatter(s['restored_best_step']/1000,s['restored_window_negative_elbo']-baseline,color=color,marker='*',s=50,zorder=5)
        inset.scatter(s['final_checked_step']/1000,windows.iloc[-1].mean_negative_elbo-baseline,color=color,marker='x',s=24,zorder=5)
    inset.set_title('After 20k steps: loss above best',fontsize=8.2)
    inset.tick_params(labelsize=7)
    inset.spines[['top','right']].set_visible(False)
    inset.grid(alpha=.15)
fig.suptitle('SHANK3 cohort 1 | Full-rank SVI convergence', fontsize=17,y=.98)
fig.legend(handles=[Line2D([],[],color='tab:blue',label='Seed 0'),Line2D([],[],color='tab:orange',label='Seed 1'),
    Line2D([],[],color='black',marker='*',ls='',ms=11,label='Restored best'),
    Line2D([],[],color='black',marker='x',ls='',ms=7,label='Final checked')],
    loc='upper center',bbox_to_anchor=(.5,.9),ncol=4,frameon=False)
fig.text(.5,.045,'1,000-step windows; ≥50,000 steps; patience 12; absolute-best checkpoint restored. '
    'Two starts, full-data float64 likelihood.\nSelection: lowest restored window negative ELBO; within a 0.1% tie, highest posterior-mean likelihood.',ha='center',fontsize=9)
for extension in ['png','pdf']:
    fig.savefig(OUT/(STEMS[1]+'.'+extension),dpi=PNG_DPI,facecolor='white')
plt.close(fig)

# %% Genotype posterior means and central 95% intervals
fig, axes = plt.subplots(1,5,figsize=(16,4.8))
fig.subplots_adjust(left=.05,right=.99,bottom=.22,top=.72,wspace=.42)
records = []
for ax, name, label in zip(axes,PARAMETERS,PARAM_LABELS):
    for position,g in enumerate(GENOTYPES):
        row = parameter_tables[g].loc[name]
        ax.errorbar(position,row['mean'],yerr=[[row['mean']-row.q025],[row.q975-row['mean']]],
            fmt='o',color=COLORS[g],capsize=5,ms=7,lw=1.8)
        records.append({'genotype':g,'parameter':name,'selected_seed':selection[g]['seed'],**row.to_dict()})
    ax.set(title=label,xticks=range(3),xticklabels=[g.upper() for g in GENOTYPES],xlim=(-.6,2.6))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.16)
fig.suptitle('SHANK3 cohort 1 | Proactive + exponential-lapse parameters',fontsize=17,y=.97)
fig.text(.5,.83,'Selected-start posterior mean and central 95% variational-posterior interval | 10,000 draws per fit',ha='center',fontsize=11)
statuses = '; '.join(f'{g.upper()}: seed {selection[g]["seed"]}, {selection[g]["status"]}' for g in GENOTYPES)
fig.text(.5,.07,'Trial-pooled genotype fits; diffusion fixed at 1; one combined proactive delay.\n'+statuses,ha='center',fontsize=10,linespacing=1.5)
for extension in ['png','pdf']:
    fig.savefig(OUT/(STEMS[2]+'.'+extension),dpi=PNG_DPI,facecolor='white')
plt.close(fig)
pd.DataFrame(records).to_csv(OUT/'selected_parameter_summary.csv',index=False)
manifest = {'source_script':str(Path(__file__).relative_to(REPO)), 'selected_starts':selected,
    'histogram_bin_width_s':BIN_WIDTH, 'theory_grid_s':MODEL_GRID_STEP,
    'denominator':'all eligible genotype trials, no area-one normalization',
    'model_censoring':'g(t)=f(t)*mean(c>t); exact bin mass=mean[F(min(b,c))-F(min(a,c))]',
    'posterior_intervals':'central 95% variational posterior; not across-animal SEM',
    'convergence_insets':'Steps >=20,000; subtract the best restored window loss within genotype for readable plateaus',
    'figures':STEMS, 'genotype_colors':COLORS, 'all_figures_single_page':True}
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(pd.DataFrame(probabilities).to_string(index=False))
