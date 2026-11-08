# %% Imports and editable source/display settings
"""Rearrange verified WT/HOM tables into two composites; no fitting or integration."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/shank3-wt-hom-composites-mpl')
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_comparison'
OUT = COMPARISON / 'composites'
FIT_ROOTS = {g: ROOT / f'asd_analysis/shank3_cohort1_{g}_gamma_omega' for g in ['wt', 'hom']}
ABLS = [20, 40, 60]
ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
ABS_ILDS = [1, 2, 4, 8, 16]
ABL_COLORS = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c'}
GROUP_COLORS = {'wt': '#1f77b4', 'hom': '#d62728'}
EXPECTED = {'wt': {'trials': 68184, 'animals': 5, 'aborts': 13750},
            'hom': {'trials': 67715, 'animals': 6, 'aborts': 13274}}
PARAMETER_SIZE = (16, 15.8)
BEHAVIOR_SIZE = (20, 17)
DPI = 180
RTD_XLIM = (-1, 1)
CHRONOMETRIC_YLIM = (0, .30)
RTD_DATA_LINEWIDTH = 1.4
RTD_MODEL_LINEWIDTH = .8
OTHER_MODEL_LINEWIDTH = 1.0
PARAMETER_STEM = 'SHANK3_cohort1_WT_HOM_all_parameters'
BEHAVIOR_STEM = 'SHANK3_cohort1_WT_HOM_behavior_comparison'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record_artist(artist, kind, **metadata):
    """Audit the actual Matplotlib values, including the rendered error-bar endpoints."""
    trace = dict(kind=kind, **metadata)
    if kind == 'stairs':
        values = artist.get_data()
        trace.update(edges=np.asarray(values.edges).tolist(), y=np.asarray(values.values).tolist(),
                     color=to_hex(artist.get_edgecolor()), linewidth=artist.get_linewidth())
    elif kind == 'scatter':
        xy = np.asarray(artist.get_offsets())
        trace.update(x=xy[:, 0].tolist(), y=xy[:, 1].tolist(),
                     color=to_hex(artist.get_facecolor()[0]))
    else:
        line = artist.lines[0] if kind == 'errorbar' else artist
        x, y = (np.asarray(a, dtype=float) for a in line.get_data())
        trace.update(x=x.tolist(), y=y.tolist(), color=to_hex(line.get_color()),
                     linewidth=line.get_linewidth(), linestyle=line.get_linestyle())
        if kind == 'errorbar':
            segments = np.asarray(artist.lines[2][0].get_segments())
            trace.update(lower=segments[:, 0, 1].tolist(), upper=segments[:, 1, 1].tolist())
    traces.append(trace)


def save_figure(fig, stem, expected_axes):
    fig.canvas.draw()
    assert len(fig.axes) == expected_axes
    panels = []
    for ax in fig.axes:
        position = ax.get_position()
        panels.append(dict(panel=ax.get_gid(), title=ax.get_title(), xlabel=ax.get_xlabel(),
            ylabel=ax.get_ylabel(), xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
            xticks=ax.get_xticks().tolist(), xticklabels=[t.get_text() for t in ax.get_xticklabels()],
            box_aspect=ax.get_box_aspect(), width_inches=position.width * fig.get_figwidth(),
            height_inches=position.height * fig.get_figheight()))
    for ext in ['png', 'pdf']:
        name = f'{stem}.{ext}'
        kwargs = {'metadata': {'Title': stem, 'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {}
        fig.savefig(OUT / name, dpi=DPI, **kwargs)
        artifacts[name] = digest(OUT / name)
    figures.append(dict(stem=stem, size_inches=fig.get_size_inches().tolist(), panels=panels))
    plt.close(fig)


# %% Preserve every prior comparison output, source CSV, and fit file
original_manifest = json.loads((COMPARISON / 'plot_manifest.json').read_text())
assert json.loads((COMPARISON / 'independent_validation.json').read_text())['status'] == 'passed'
baseline_file = OUT / 'preservation_baseline.json'
if not baseline_file.exists():
    baseline = json.loads((COMPARISON / 'preservation_baseline.json').read_text())
    paths = [p for p in COMPARISON.iterdir() if p.is_file()]
    paths += list((ROOT / 'docs/assets/results/2026-10-23/shank3_cohort1_wt_hom_comparison').glob('*.*'))
    paths += [ROOT / 'asd_analysis' / f'{prefix}_shank3_cohort1_wt_hom_comparison.py'
              for prefix in ['plot', 'verify', 'publish']]
    baseline.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    baseline_file.write_text(json.dumps(baseline, indent=2) + '\n')
baseline = json.loads(baseline_file.read_text())
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name

parameters = pd.read_csv(COMPARISON / 'parameter_comparisons.csv', float_precision='round_trip')
chronometrics = pd.read_csv(COMPARISON / 'chronometric_points.csv', float_precision='round_trip')
rtd_data, rtd_model, psychometrics = {}, {}, {}
for group, folder in FIT_ROOTS.items():
    print('Saved condition-delay source:', folder, flush=True)
    rtd_data[group] = pd.read_csv(folder / 'rtd_data_bins.csv', float_precision='round_trip')
    rtd_model[group] = pd.read_csv(folder / 'condition_delay/pooled_rtd_model_1ms.csv.gz', float_precision='round_trip')
    psychometrics[group] = pd.read_csv(folder / 'condition_delay/psychometric_predictions.csv', float_precision='round_trip')
    assert int(psychometrics[group].trials.sum()) == EXPECTED[group]['trials']
assert len(parameters) == 184 and len(chronometrics) == 40
traces, figures, artifacts = [], [], {}

# %% All 92 reactive parameters per genotype in eleven panels
fig = plt.figure(figsize=PARAMETER_SIZE)
grid = fig.add_gridspec(4, 6, left=.075, right=.975, top=.90, bottom=.075, hspace=.62, wspace=1.0)
for row, (block, label) in enumerate([
        ('gamma', 'Gamma (dimensionless)'), ('omega', 'Omega (s⁻¹)'),
        ('t_E_aff', 'Reactive delay (ms)')]):
    for column, abl in enumerate(ABLS):
        ax = fig.add_subplot(grid[row, 2 * column:2 * column + 2])
        panel = f'{block}_{abl}'
        ax.set_gid(panel)
        for group, offset in [('wt', -.10), ('hom', .10)]:
            z = parameters.loc[parameters.group.eq(group) & parameters.block.eq(block) & parameters.ABL.eq(abl)].sort_values('ILD')
            assert z.ILD.tolist() == ILDS
            y = z.display_mean.to_numpy()
            error = np.vstack([y - z.display_lower95, z.display_upper95 - y])
            artist = ax.errorbar(np.arange(10) + offset, y, yerr=error, fmt='o', ms=4.5,
                color=GROUP_COLORS[group], capsize=3, elinewidth=1, linestyle='none')
            record_artist(artist, 'errorbar', figure=PARAMETER_STEM, panel=panel, role='posterior',
                          group=group, ABL=abl, block=block, ILDs=ILDS)
        ax.set(title=f'ABL {abl} dB', xlabel='Signed ILD (dB)',
               xticks=np.arange(10), xticklabels=ILDS, xlim=(-.5, 9.5),
               ylim=original_manifest['parameter_ylim'][block])
        ax.tick_params(axis='x', labelrotation=45)
        if column == 0:
            ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.18)

for block, label, columns in [('w', 'Starting bias w (dimensionless)', slice(1, 3)),
                             ('del_go', 'Readout delay del_go (ms)', slice(3, 5))]:
    ax = fig.add_subplot(grid[3, columns])
    ax.set_gid(block)
    for i, group in enumerate(['wt', 'hom']):
        z = parameters.loc[parameters.group.eq(group) & parameters.block.eq(block)].iloc[0]
        artist = ax.errorbar([i], [z.display_mean],
            yerr=[[z.display_mean - z.display_lower95], [z.display_upper95 - z.display_mean]],
            fmt='o', color=GROUP_COLORS[group], ms=6, capsize=4, elinewidth=1, linestyle='none')
        record_artist(artist, 'errorbar', figure=PARAMETER_STEM, panel=block, role='posterior',
                      group=group, ABL=None, block=block)
    ax.set(title='Shared across all signed conditions', ylabel=label,
           xticks=[0, 1], xticklabels=['WT', 'HOM'], xlim=(-.5, 1.5),
           ylim=original_manifest['parameter_ylim'][block])
    ax.grid(axis='y', alpha=.18)

fig.suptitle('SHANK3 cohort 1 | WT–HOM reactive parameters', fontsize=18, y=.978)
fig.text(.5, .950, 'WT: 5 animals / 68,184 choices     •     HOM: 6 animals / 67,715 choices',
         ha='center', fontsize=11)
fig.legend(handles=[Line2D([], [], color=GROUP_COLORS[g], marker='o', linestyle='',
           label=g.upper()) for g in ['wt', 'hom']], loc='upper center', bbox_to_anchor=(.5, .936),
           ncol=2, frameon=False, fontsize=11)
fig.text(.5, .023, 'Dots: posterior means. Bars: central 95% credible intervals from 10,000 saved draws.\n'
         '30 Gamma + 30 Omega + 30 reactive delays + shared w and del_go per genotype; no ±ILD symmetry.\n'
         'Conditional variational uncertainty; five proactive/lapse parameters remain frozen. No refitting.',
         ha='center', va='bottom', fontsize=9)
save_figure(fig, PARAMETER_STEM, 11)

# %% Behavior: two RTD rows, three compact chronometrics, two psychometrics
fig = plt.figure(figsize=BEHAVIOR_SIZE)
rows = fig.add_gridspec(4, 1, left=.065, right=.98, top=.90, bottom=.085,
                       height_ratios=[1, 1, 1.05, 1.05], hspace=.65)
rtd_ymax = 1.08 * max(max(z.density.max() for z in rtd_data.values()),
                       max(z.density.max() for z in rtd_model.values()))
for row, group in enumerate(['wt', 'hom']):
    grid = rows[row].subgridspec(1, 5, wspace=.31)
    for column, abs_ild in enumerate(ABS_ILDS):
        ax = fig.add_subplot(grid[0, column])
        panel = f'{group}_rtd_{abs_ild}'
        ax.set_gid(panel)
        counts = []
        for abl in ABLS:
            z = rtd_data[group].loc[rtd_data[group].ABL.eq(abl) & rtd_data[group].abs_ILD.eq(abs_ild)].sort_values('bin')
            model = rtd_model[group].loc[rtd_model[group].ABL.eq(abl) & rtd_model[group].abs_ILD.eq(abs_ild)].sort_values('rt_s')
            line, = ax.plot(model.rt_s, model.density, color=ABL_COLORS[abl], linestyle='--',
                            linewidth=RTD_MODEL_LINEWIDTH, zorder=2)
            record_artist(line, 'line', figure=BEHAVIOR_STEM, panel=panel, role='rtd_model',
                          group=group, ABL=abl, abs_ILD=abs_ild)
            edges = np.r_[z.left.to_numpy(), z.right.iloc[-1]]
            stairs = ax.stairs(z.density.to_numpy(), edges, color=ABL_COLORS[abl],
                               linewidth=RTD_DATA_LINEWIDTH, fill=False, baseline=None, zorder=3)
            record_artist(stairs, 'stairs', figure=BEHAVIOR_STEM, panel=panel, role='rtd_data',
                          group=group, ABL=abl, abs_ILD=abs_ild)
            counts.append(int(z['count'].sum()))
        ax.axvline(0, color='.65', linewidth=.6, zorder=0)
        ax.set(title=f'|ILD| = {abs_ild} dB\nn₂₀/₄₀/₆₀ = ' + '/'.join(f'{n:,}' for n in counts),
               xlabel='Stimulus-relative RT (s)', xlim=RTD_XLIM, ylim=(0, rtd_ymax),
               xticks=[-1, -.5, 0, .5, 1])
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=9)
        if column == 0:
            ax.set_ylabel('RT density (s⁻¹)')
        else:
            ax.tick_params(labelleft=False)
        ax.grid(axis='y', alpha=.14)
    top = rows[row].get_position(fig).y1
    n = EXPECTED[group]
    fig.text(.065, top + .035, f'{"ab"[row]}   {group.upper()} RTDs | {n["animals"]} animals; '
             f'{n["trials"] + n["aborts"]:,} observations = {n["trials"]:,} valid + {n["aborts"]:,} pre-onset aborts',
             fontsize=12, weight='bold')

grid = rows[2].subgridspec(1, 3, wspace=.55)
for column, group in enumerate(['wt', 'hom', None]):
    ax = fig.add_subplot(grid[0, column])
    panel = 'chronometric_pooled' if group is None else f'{group}_chronometric'
    ax.set_gid(panel)
    ax.set_box_aspect(1)
    curves = [(g, None, GROUP_COLORS[g], offset) for g, offset in [('wt', -.04), ('hom', .04)]] if group is None else [
        (group, abl, ABL_COLORS[abl], offset) for abl, offset in zip(ABLS, [-.06, 0, .06])]
    for g, abl, color, offset in curves:
        z = chronometrics.loc[chronometrics.group.eq(g) &
            chronometrics.view.eq('ABL_pooled' if group is None else 'ABL_separate')]
        if abl is not None:
            z = z.loc[z.ABL.eq(abl)]
        z = z.sort_values('abs_ILD')
        x = np.arange(5) + offset
        line, = ax.plot(x, z.model_mean_RT_s, color=color, linewidth=OTHER_MODEL_LINEWIDTH, zorder=2)
        record_artist(line, 'line', figure=BEHAVIOR_STEM, panel=panel, role='chronometric_model', group=g, ABL=abl)
        bars = ax.errorbar(x, z.data_mean_RT_s, yerr=z.animal_SEM_s, fmt='o', color=color,
                          ms=4.8, capsize=3, elinewidth=1, linestyle='none', zorder=3)
        record_artist(bars, 'errorbar', figure=BEHAVIOR_STEM, panel=panel, role='chronometric_data', group=g, ABL=abl)
    title = 'WT vs HOM | ABLs pooled' if group is None else f'{group.upper()} | three ABLs'
    subtitle = 'WT n = 68,184; HOM n = 67,715' if group is None else f'{EXPECTED[group]["animals"]} animals; n = {EXPECTED[group]["trials"]:,}'
    ax.set(title=title + '\n' + subtitle, xlabel='|ILD| (dB)', ylabel='Mean RT (s)',
           xticks=np.arange(5), xticklabels=ABS_ILDS, xlim=(-.35, 4.35), ylim=CHRONOMETRIC_YLIM)
    ax.title.set_fontsize(10)
    ax.grid(axis='y', alpha=.14)
    if group is None:
        ax.legend(handles=[Line2D([], [], color=GROUP_COLORS[g], label=g.upper()) for g in ['wt', 'hom']],
                  loc='lower right', fontsize=8, frameon=False)
fig.text(.065, rows[2].get_position(fig).y1 + .035,
         'c   Chronometrics | dots: pooled mean ± SEM across animals; thin solid lines: conditional model mean',
         fontsize=12, weight='bold')

grid = rows[3].subgridspec(1, 4, width_ratios=[.6, 1, 1, .6], wspace=.45)
for column, group in enumerate(['wt', 'hom'], start=1):
    ax = fig.add_subplot(grid[0, column])
    panel = f'{group}_psychometric'
    ax.set_gid(panel)
    ax.set_box_aspect(1)
    for abl in ABLS:
        z = psychometrics[group].loc[psychometrics[group].ABL.eq(abl)].sort_values('ILD')
        line, = ax.plot(z.ILD, z.model_p_right, color=ABL_COLORS[abl], linewidth=OTHER_MODEL_LINEWIDTH, zorder=2)
        record_artist(line, 'line', figure=BEHAVIOR_STEM, panel=panel, role='psychometric_model', group=group, ABL=abl)
        dots = ax.scatter(z.ILD, z.data_p_right, color=ABL_COLORS[abl], s=22, zorder=3)
        record_artist(dots, 'scatter', figure=BEHAVIOR_STEM, panel=panel, role='psychometric_data', group=group, ABL=abl)
    ax.set(title=f'{group.upper()} | {EXPECTED[group]["animals"]} animals; n = {EXPECTED[group]["trials"]:,}',
           xlabel='Signed ILD (dB)', ylabel='P(right)', xlim=(-18, 18), ylim=(0, 1),
           xticks=[-16, -8, 0, 8, 16], yticks=[0, .25, .5, .75, 1])
    ax.title.set_fontsize(10)
    ax.grid(alpha=.14)
fig.text(.065, rows[3].get_position(fig).y1 + .035,
         'd   Psychometrics | measured points and model predictions at the ten fitted signed ILDs',
         fontsize=12, weight='bold')

fig.suptitle('SHANK3 cohort 1 | WT–HOM model and data', fontsize=18, y=.983)
handles = [Line2D([], [], color=ABL_COLORS[a], linewidth=1.5, label=f'ABL {a} dB') for a in ABLS]
handles += [Line2D([], [], color='.25', linewidth=RTD_DATA_LINEWIDTH, label='RTD data: solid steps'),
            Line2D([], [], color='.25', linewidth=RTD_MODEL_LINEWIDTH, linestyle='--', label='RTD model: thin dashed')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965), ncol=5, frameon=False, fontsize=10)
fig.text(.5, .025, 'Session type 1; training 16; ABL 59→60; prepared exclusions retained. '
         'Valid choices: success ∈ {1,−1}, recorded RT ∈ [0,1] s.\n'
         'RTDs additionally show pre-onset aborts in [−1,0); 20 ms bins; each genotype/ABL/|ILD| data and model curve has area 1 in [−1,1] s.\n'
         'Chronometric dots pool trials; SEM weights animals equally. Model means normalize at each exact valid-trial onset before averaging.\n'
         'RTD models use parent-trial onset/sign weights; chronometrics and psychometrics use fitted-valid-trial weights. No refitting or changed values.',
         ha='center', va='bottom', fontsize=8.7)
save_figure(fig, BEHAVIOR_STEM, 15)

# %% Save actual plotted arrays, panel geometry and provenance
with gzip.open(OUT / 'plotted_series.json.gz', 'wt') as stream:
    json.dump(traces, stream, allow_nan=False)
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name
manifest = dict(status='generated_pending_independent_verification', figures=figures, artifacts=artifacts,
    groups=EXPECTED, parameter_estimates=184, chronometric_pairs=40, psychometric_pairs=60,
    rtd_data_curves=30, rtd_model_curves=30, group_colors=GROUP_COLORS, ABL_colors=ABL_COLORS,
    rtd_xlim=RTD_XLIM, rtd_ylim=[0, rtd_ymax], chronometric_ylim=CHRONOMETRIC_YLIM,
    rtd_data_linewidth=RTD_DATA_LINEWIDTH, rtd_model_linewidth=RTD_MODEL_LINEWIDTH,
    other_model_linewidth=OTHER_MODEL_LINEWIDTH, plotted_series=len(traces),
    input_fit_and_previous_export_hashes_unchanged=True, refitting=False,
    source=str(Path(__file__).relative_to(ROOT)), source_sha256=digest(Path(__file__)),
    plotted_series_sha256=digest(OUT / 'plotted_series.json.gz'))
(OUT / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Saved two composites and actual-artist numerical audits:', OUT, flush=True)
