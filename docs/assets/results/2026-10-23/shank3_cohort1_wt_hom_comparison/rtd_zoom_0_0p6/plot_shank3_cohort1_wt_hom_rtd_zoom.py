# %% Editable sources and display settings (no fitting, binning, or normalization)
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/shank3-rtd-zoom-mpl')
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_comparison'
OUT = COMPARISON / 'rtd_zoom_0_0p6'
FIT_ROOTS = {g: ROOT / f'asd_analysis/shank3_cohort1_{g}_gamma_omega' for g in ['wt', 'hom']}
XLIM = (0, .6)
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
COLORS = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c'}
DATA_LINEWIDTH = 1.4
MODEL_LINEWIDTH = .7
FIGSIZE = (18, 10)
DPI = 180
COUNTS = {'wt': (5, 68184, 13750), 'hom': (6, 67715, 13274)}
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# %% Reuse the full, previously verified arrays and the latest composite's y-scale
previous = COMPARISON / 'composites'
previous_manifest = json.loads((previous / 'plot_manifest.json').read_text())
assert json.loads((previous / 'independent_validation.json').read_text())['status'] == 'passed'
YLIM = previous_manifest['rtd_ylim']
with gzip.open(previous / 'plotted_series.json.gz', 'rt') as stream:
    previous_series = json.load(stream)
baseline = json.loads((previous / 'preservation_baseline.json').read_text())
baseline.update({str(p.relative_to(ROOT)): digest(p) for p in previous.iterdir() if p.is_file()})
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name
artifacts, panels = {}, []

for group, source in FIT_ROOTS.items():
    print('Saved condition-delay RTDs:', source, flush=True)
    data = pd.read_csv(source / 'rtd_data_bins.csv', float_precision='round_trip')
    model = pd.read_csv(source / 'condition_delay/pooled_rtd_model_1ms.csv.gz', float_precision='round_trip')
    fig, axes = plt.subplots(3, 5, figsize=FIGSIZE, sharex=True, sharey=True)
    for row, abl in enumerate(ABLS):
        for column, abs_ild in enumerate(ABS_ILDS):
            ax = axes[row, column]
            z = data.loc[data.ABL.eq(abl) & data.abs_ILD.eq(abs_ild)].sort_values('bin')
            curve = model.loc[model.ABL.eq(abl) & model.abs_ILD.eq(abs_ild)].sort_values('rt_s')
            edges = np.r_[z.left.to_numpy(), z.right.iloc[-1]]
            line, = ax.plot(curve.rt_s, curve.density, color='black', linewidth=MODEL_LINEWIDTH, zorder=2)
            steps = ax.stairs(z.density.to_numpy(), edges, baseline=None, fill=False,
                              color=COLORS[abl], linewidth=DATA_LINEWIDTH, zorder=3)
            ax.set(xlim=XLIM, ylim=YLIM, xticks=[0, .1, .2, .3, .4, .5, .6],
                   title=f'|ILD| = {abs_ild} dB  ·  n_full = {int(z.display_trials.iloc[0]):,}')
            ax.title.set_fontsize(10)
            if column == 0:
                ax.set_ylabel(f'ABL {abl} dB\nDensity (s⁻¹)')
            if row == 2:
                ax.set_xlabel('RT from stimulus onset (s)')
            ax.grid(axis='y', alpha=.15)

            # Compare actual new artists to the exact artists in the prior composite.
            for role, actual_x, actual_y in [
                    ('rtd_data', steps.get_data().edges, steps.get_data().values),
                    ('rtd_model', line.get_xdata(), line.get_ydata())]:
                original = [t for t in previous_series if t['role'] == role and t['group'] == group
                            and t['ABL'] == abl and t['abs_ILD'] == abs_ild]
                assert len(original) == 1
                np.testing.assert_array_equal(actual_x, original[0]['edges' if role == 'rtd_data' else 'x'])
                np.testing.assert_array_equal(actual_y, original[0]['y'])
            assert list(ax.get_xlim()) == list(XLIM) and list(ax.get_ylim()) == YLIM
            assert line.get_linewidth() == MODEL_LINEWIDTH < steps.get_linewidth()
            assert line.get_zorder() < steps.get_zorder()
            panels.append(dict(group=group, ABL=abl, abs_ILD=abs_ild,
                xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), full_window_trials=int(z.display_trials.iloc[0]),
                data_bins_unchanged=len(z), model_samples_unchanged=len(curve)))

    animals, choices, aborts = COUNTS[group]
    fig.suptitle(f'SHANK3 cohort 1 {group.upper()} | condition-specific-delay model\n'
                 f'{animals} animals; display zoom: 0–0.6 s', fontsize=15, y=.985)
    fig.legend(handles=[Line2D([], [], color=COLORS[a], linewidth=DATA_LINEWIDTH, label=f'Data: ABL {a} dB') for a in ABLS]
               + [Line2D([], [], color='black', linewidth=MODEL_LINEWIDTH, label='Model (thin line)')],
               loc='upper center', bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
    fig.text(.5, .024, 'Display zoom only: saved 20 ms data bins and 1 ms model values are unchanged; no recalculation or renormalization.\n'
             f'Full [−1,1] s normalization includes {choices:,} valid choices + {aborts:,} pre-onset aborts; n_full labels count that full window.\n'
             'Each panel pools actual ±ILD trials. The visible 0–0.6 s segment is not normalized to area 1.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.065, right=.99, top=.86, bottom=.13, hspace=.32, wspace=.16)
    assert len(fig.axes) == 15
    stem = f'SHANK3_cohort1_{group.upper()}_condition_delay_RTDs_xlim_0_0p6s'
    for ext in ['png', 'pdf']:
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=DPI)
        artifacts[path.name] = digest(path)
        if ext == 'pdf':
            info = subprocess.check_output(['pdfinfo', str(path)], text=True)
            assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:'))) == 1
    plt.close(fig)

# %% Preserve source files and record the display-only checks
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name
report = dict(status='passed', figures=2, panels=panels, artifacts=artifacts,
    displayed_xlim=list(XLIM), unchanged_ylim=YLIM,
    data_linewidth=DATA_LINEWIDTH, model_linewidth=MODEL_LINEWIDTH,
    data_drawn_above_model=True, data_bins_unchanged=3000, model_samples_unchanged=60030,
    normalization_window_s=[-1, 1], renormalized=False, refitted=False,
    all_new_artist_values_equal_previous_artists=True, single_page_PDFs=2,
    preserved_files=len(baseline), source=str(Path(__file__).relative_to(ROOT)), source_sha256=digest(Path(__file__)))
(OUT / 'display_validation.json').write_text(json.dumps(report, indent=2) + '\n')
(OUT / 'preservation_baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
print('Saved and checked both 3 × 5 zoomed RTD figures:', OUT, flush=True)
