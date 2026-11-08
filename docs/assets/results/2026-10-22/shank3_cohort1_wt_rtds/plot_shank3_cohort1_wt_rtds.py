# %% Imports
"""SHANK3 cohort 1 WT: pooled-trial RTDs, five |ILD| panels and three ABLs.

Read the cleaned prepared CSV; its preparation exclusions are already applied.
Run from the repository root with .venv/bin/python.
"""
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / 'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
OUTPUT = REPO / 'asd_analysis/shank3_cohort1_wt_rtds'
FILENAME = 'SHANK3_cohort1.csv'
GENOTYPE = 'wt'
ABS_ILDS = [1, 2, 4, 8, 16]
ABLS = [20, 40, 60]
COLORS = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
RT_WINDOW = (-1., 1.)
BIN_MS = 20
FIGSIZE = (20, 4.8)
PNG_DPI = 220
NAME = 'SHANK3_cohort1_WT_pooled_rtds_by_abs_ILD_minus1_to1'
OUTPUT.mkdir(parents=True, exist_ok=True)

# %% Select recorded RTs; no further repeat filtering
with INPUT.open('rb') as stream:
    input_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
columns = ['filename', 'animal', 'genotype', 'success', 'abort_type', 'timed_rt',
           'ABL', 'ILD', 'session_type', 'training_level', 'trial_is_repeat', 'is_short_sound']
data = pd.read_csv(INPUT, usecols=columns, float_precision='round_trip')
cohort = data.loc[data.filename.eq(FILENAME) & data.genotype.eq(GENOTYPE)].copy()
assert cohort.animal.nunique() == 5
assert cohort.session_type.eq(1).all() and cohort.training_level.eq(16).all()
assert cohort.trial_is_repeat.eq(False).all() and cohort.is_short_sound.eq(0).all()
cohort['plot_ABL'] = cohort.ABL.replace({59: 60})
selected = cohort.loc[(cohort.success.isin([1, -1]) | cohort.abort_type.eq('Fixation'))
    & np.isfinite(cohort.timed_rt) & cohort.timed_rt.between(*RT_WINDOW)
    & cohort.plot_ABL.isin(ABLS) & cohort.ILD.abs().isin(ABS_ILDS)].copy()
selected['abs_ILD'] = selected.ILD.abs().astype(int)
selected['prepared_csv_line'] = selected.index + 2
assert len(selected) == 83089 and selected.animal.nunique() == 5
assert selected.success.isin([1, -1]).sum() == 68184
assert selected.timed_rt.lt(0).sum() == 14820
selected.to_csv(OUTPUT / 'selected_trials.csv.gz', index=False, compression='gzip')
membership = selected.groupby('animal', as_index=False).agg(
    trials=('animal', 'size'), negative_RT_trials=('timed_rt', lambda x: int((x < 0).sum())))
membership.to_csv(OUTPUT / 'animal_membership.csv', index=False)
edges = np.arange(int(RT_WINDOW[0]*1000), int(RT_WINDOW[1]*1000)+1, BIN_MS)/1000
assert edges[0] == -1 and edges[-1] == 1 and 0 in edges
np.testing.assert_allclose(np.diff(edges), .020, atol=1e-15, rtol=0)

# %% Pool trials directly within ABL and |ILD|, then normalize each curve to area 1
curves, histograms, counts = {}, [], []
for ild in ABS_ILDS:
    for abl in ABLS:
        rows = selected.loc[selected.abs_ILD.eq(ild) & selected.plot_ABL.eq(abl)]
        hist = np.histogram(rows.timed_rt, bins=edges)[0]
        n = len(rows)
        density = hist/(n*np.diff(edges))
        assert hist.sum() == n and np.isclose(np.sum(density*np.diff(edges)), 1)
        curves[(ild, abl)] = (density, n)
        counts.append({'abs_ILD': ild, 'ABL': abl, 'animals': rows.animal.nunique(),
            'trials': n, 'valid_choices': int(rows.success.isin([1, -1]).sum()),
            'fixation_aborts': int((~rows.success.isin([1, -1]) & rows.abort_type.eq('Fixation')).sum()),
            'negative_RT_trials': int(rows.timed_rt.lt(0).sum()), 'area': float(np.sum(density*np.diff(edges)))})
        histograms.append(pd.DataFrame({'abs_ILD': ild, 'ABL': abl, 'bin_index': np.arange(len(hist)),
            'bin_left_s': edges[:-1], 'bin_right_s': edges[1:], 'bin_count': hist,
            'normalization_trials': n, 'density_per_s': density}))
counts = pd.DataFrame(counts)
bins = pd.concat(histograms, ignore_index=True)
assert counts.trials.sum() == len(selected) and counts.animals.eq(5).all()
counts.to_csv(OUTPUT / 'condition_counts.csv', index=False)
bins.to_csv(OUTPUT / 'histogram_bins.csv', index=False)

# %% Independently read CSV cells and reconstruct identities and histograms
independent = Counter()
independent_lines = []
independent_animals = set()
with INPUT.open(newline='') as stream:
    for line, row in enumerate(csv.DictReader(stream), start=2):
        if row['filename'] != FILENAME or row['genotype'] != GENOTYPE:
            continue
        valid = row['success'] in {'1', '-1'}
        if not (valid or row['abort_type'] == 'Fixation'):
            continue
        rt, abl, ild = [float(row[c] or 'nan') for c in ['timed_rt', 'ABL', 'ILD']]
        abl = 60 if abl == 59 else abl
        if not (np.isfinite(rt) and -1 <= rt <= 1 and abl in ABLS and abs(ild) in ABS_ILDS):
            continue
        index = min(len(edges)-2, bisect_right(edges.tolist(), rt)-1)
        independent[(int(abs(ild)), int(abl), index)] += 1
        independent_lines.append(line)
        independent_animals.add(row['animal'])
np.testing.assert_array_equal(independent_lines, selected.prepared_csv_line)
assert independent_animals == set(selected.animal)
for row in bins.itertuples(index=False):
    assert row.bin_count == independent[(row.abs_ILD, row.ABL, row.bin_index)]
    assert np.isclose(row.density_per_s, row.bin_count/(row.normalization_trials*(row.bin_right_s-row.bin_left_s)))

# %% Five |ILD| panels, ABL colors and common axes
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True, sharey=True)
fig.subplots_adjust(left=.045, right=.99, bottom=.22, top=.73, wspace=.12)
for ax, ild in zip(axes, ABS_ILDS):
    for abl in ABLS:
        density, n = curves[(ild, abl)]
        ax.stairs(density, edges, color=COLORS[abl], linewidth=1.5,
            label=f'{abl} dB (n={n:,})')
    ax.axvline(0, color='.45', linestyle=':', linewidth=.9, zorder=0)
    ax.set(title=f'|ILD| = {ild} dB', xlim=RT_WINDOW, ylim=(0, max(d.max() for d, n in curves.values())*1.07))
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel('RT from stimulus onset (s)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
axes[0].set_ylabel('Probability density (s⁻¹)')
fig.suptitle('SHANK3 cohort 1 | WT | five animals pooled', fontsize=17, y=.97)
fig.text(.5, .85, 'Valid choices (success ±1) + fixation aborts | 83,089 trials | 20 ms bins', ha='center', fontsize=12)
fig.text(.5, .07, 'Actual positive/negative ILD trials pooled across animals and sessions. ABL 59→60.\n'
    'Each ABL/|ILD| curve is normalized over the full [−1,1] s window: count / (condition trials × bin width).',
    ha='center', fontsize=10, linespacing=1.5)
fig.savefig(OUTPUT / f'{NAME}.png', dpi=PNG_DPI, facecolor='white')
fig.savefig(OUTPUT / f'{NAME}.pdf')
plt.close(fig)
with INPUT.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == input_sha
report = {'status': 'passed', 'source': str(INPUT.relative_to(REPO)), 'source_sha256': input_sha,
    'plotter': str(Path(__file__).relative_to(REPO)), 'filename': FILENAME, 'genotype': GENOTYPE,
    'animal_ids': sorted(independent_animals), 'animals': 5, 'selected_trials': len(selected),
    'valid_choices': 68184, 'fixation_aborts': 14905, 'negative_RT_trials': 14820,
    'RT_window_s': list(RT_WINDOW), 'bin_ms': BIN_MS, 'bins_per_curve': len(edges)-1,
    'curves': len(counts), 'all_unit_areas': True, 'all_condition_counts_independently_checked': True,
    'all_trial_identities_independently_checked': True, 'input_unchanged': True,
    'pooling': 'Actual trials pooled within ABL and abs(ILD); no averaging of animal densities',
    'selection': 'Cleaned session1 CSV; WT, SHANK3 cohort1; success 1/-1 or Fixation; finite RT in inclusive [-1,1]; ABL59->60 then20/40/60; absILD1/2/4/8/16',
    'prepared_repeat_filter_reapplied': False}
(OUTPUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
