# %% Imports
"""Pooled fixation-abort durations for SHANK3 cohort 1 WT, HOM and HET.

Select negative stimulus-relative RTs, but histogram recorded fix_time in
seconds. This is a different timing coordinate from timed_rt.
Run with .venv/bin/python from the repository root.
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
OUTPUT = REPO / 'asd_analysis/shank3_cohort1_fixation_abort_durations'
FILENAME = 'SHANK3_cohort1.csv'
GENOTYPES = ['wt', 'hom', 'het']
LABELS = {'wt': 'WT', 'hom': 'HOM', 'het': 'HET'}
COLORS = {'wt': 'tab:blue', 'hom': 'tab:green', 'het': 'tab:orange'}
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
BIN_MS = 20
FIGSIZE = (13, 4.7)
PNG_DPI = 240
NAME = 'SHANK3_cohort1_fixation_abort_duration_WT_HOM_HET'
OUTPUT.mkdir(parents=True, exist_ok=True)

# %% Preserve preparation rules; only negative timed_rt selects trials
with INPUT.open('rb') as stream:
    input_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
cols = ['filename', 'animal', 'genotype', 'abort_type', 'success', 'timed_rt', 'fix_time',
        'ABL', 'ILD', 'session_type', 'training_level', 'trial_is_repeat', 'is_short_sound']
data = pd.read_csv(INPUT, usecols=cols, float_precision='round_trip')
cohort = data.loc[data.filename.eq(FILENAME) & data.genotype.isin(GENOTYPES)].copy()
assert cohort.session_type.eq(1).all() and cohort.training_level.eq(16).all()
assert cohort.trial_is_repeat.eq(False).all() and cohort.is_short_sound.eq(0).all()
cohort['plot_ABL'] = cohort.ABL.replace({59: 60})
selected = cohort.loc[cohort.abort_type.eq('Fixation') & np.isfinite(cohort.timed_rt) & cohort.timed_rt.lt(0)
    & cohort.plot_ABL.isin(ABLS) & cohort.ILD.abs().isin(ABS_ILDS)].copy()
assert np.isfinite(selected.fix_time).all() and selected.fix_time.ge(0).all()
selected['fixation_duration_s'] = selected.fix_time/1000
selected['prepared_csv_line'] = selected.index+2
assert len(selected) == 39406 and selected.animal.nunique() == 14
assert selected.timed_rt.lt(-1).any()  # No inherited lower RT cutoff.
selected.to_csv(OUTPUT/'selected_trials.csv.gz', index=False, compression='gzip')
membership = selected.groupby(['genotype', 'animal'], as_index=False).agg(
    trials=('animal', 'size'), mean_duration_s=('fixation_duration_s', 'mean'))
membership.to_csv(OUTPUT/'animal_membership.csv', index=False)
# One shared range covers every finite measured duration, with no upper truncation.
end_ms = int(np.ceil(selected.fix_time.max()/100)*100)
edges = np.arange(0, end_ms+1, BIN_MS)/1000
assert edges[-1] >= selected.fixation_duration_s.max()
np.testing.assert_allclose(np.diff(edges), BIN_MS/1000, atol=1e-15, rtol=0)

# %% Pool observed trials across animals, sessions, ABLs and signed ILDs
curves, records, bin_tables = {}, [], []
for genotype in GENOTYPES:
    rows = selected.loc[selected.genotype.eq(genotype)]
    counts = np.histogram(rows.fixation_duration_s, bins=edges)[0]
    n = len(rows)
    density = counts/(n*(BIN_MS/1000))
    assert counts.sum() == n
    assert np.isclose(np.sum(density*np.diff(edges)), 1)
    assert (rows.animal.nunique(), n) == {'wt': (5, 15279), 'hom': (6, 13580), 'het': (3, 10547)}[genotype]
    curves[genotype] = density
    records.append({'genotype': genotype, 'animals': rows.animal.nunique(), 'trials': n,
        'timed_rt_below_minus1_trials': int(rows.timed_rt.lt(-1).sum()),
        'mean_fixation_duration_s': rows.fixation_duration_s.mean(),
        'median_fixation_duration_s': rows.fixation_duration_s.median(),
        'max_fixation_duration_s': rows.fixation_duration_s.max(),
        'density_area': float(np.sum(density*np.diff(edges)))})
    bin_tables.append(pd.DataFrame({'genotype': genotype, 'bin_index': np.arange(len(counts)),
        'bin_left_s': edges[:-1], 'bin_right_s': edges[1:], 'bin_count': counts,
        'normalization_trials': n, 'density_per_s': density}))
summary = pd.DataFrame(records)
bins = pd.concat(bin_tables, ignore_index=True)
summary.to_csv(OUTPUT/'genotype_counts.csv', index=False)
bins.to_csv(OUTPUT/'histogram_bins.csv', index=False)

# %% Independent CSV-cell parsing, trial identities and scalar bin assignments
independent_counts = Counter()
independent_lines, independent_durations = [], []
with INPUT.open(newline='') as stream:
    for line, row in enumerate(csv.DictReader(stream), start=2):
        if row['filename'] != FILENAME or row['genotype'] not in GENOTYPES or row['abort_type'] != 'Fixation':
            continue
        rt, abl, ild = [float(row[k] or 'nan') for k in ['timed_rt', 'ABL', 'ILD']]
        abl = 60 if abl == 59 else abl
        if not (np.isfinite(rt) and rt < 0 and abl in ABLS and abs(ild) in ABS_ILDS):
            continue
        duration = float(row['fix_time'])/1000
        assert np.isfinite(duration) and 0 <= duration <= edges[-1]
        index = min(len(edges)-2, bisect_right(edges.tolist(), duration)-1)
        independent_counts[(row['genotype'], index)] += 1
        independent_lines.append(line)
        independent_durations.append(duration)
np.testing.assert_array_equal(independent_lines, selected.prepared_csv_line)
np.testing.assert_array_equal(independent_durations, selected.fixation_duration_s)
for row in bins.itertuples(index=False):
    assert row.bin_count == independent_counts[(row.genotype, row.bin_index)]
    assert np.isclose(row.density_per_s, row.bin_count/(row.normalization_trials*(row.bin_right_s-row.bin_left_s)))

# %% One panel per genotype, shared scales and unfilled step distributions
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
fig.subplots_adjust(left=.065, right=.985, bottom=.27, top=.71, wspace=.13)
for ax, genotype in zip(axes, GENOTYPES):
    count = summary.loc[summary.genotype.eq(genotype)].iloc[0]
    ax.stairs(curves[genotype], edges, color=COLORS[genotype], linewidth=1.8)
    ax.set(title=f'{LABELS[genotype]} | {int(count.animals)} animals | n={int(count.trials):,}',
        xlim=(0, edges[-1]), ylim=(0, max(c.max() for c in curves.values())*1.08))
    ax.set_xticks(np.arange(0, edges[-1]+.001, .5))
    ax.set_xlabel('Recorded fixation duration (s)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Probability density (s⁻¹)')
fig.suptitle('SHANK3 cohort 1 | Fixation-abort durations', fontsize=17, y=.97)
fig.text(.5, .84, 'Fixation aborts with timed_rt < 0, no lower RT cutoff | 39,406 trials | 20 ms bins', ha='center', fontsize=11)
fig.text(.5, .065, 'Histogrammed quantity: recorded fix_time / 1000 (seconds). ABLs and signed ILDs pooled by trial count.\n'
    'ABL 20/40/60 (59→60); |ILD| 1/2/4/8/16. Each genotype curve has area 1; all selected durations are included.',
    ha='center', fontsize=9.8, linespacing=1.5)
fig.savefig(OUTPUT/(NAME+'.png'), dpi=PNG_DPI, facecolor='white')
fig.savefig(OUTPUT/(NAME+'.pdf'))
plt.close(fig)
with INPUT.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == input_sha
report = {'status': 'passed', 'source': str(INPUT.relative_to(REPO)), 'source_sha256': input_sha,
    'plotter': str(Path(__file__).relative_to(REPO)), 'filename': FILENAME,
    'plotted_coordinate': 'Recorded fix_time in milliseconds / 1000, expressed in seconds',
    'selection': 'Cleaned session1 CSV; abort_type=Fixation (ASD equivalent of abort_event=3); finite timed_rt<0 with no lower cutoff; ABL59->60 then20/40/60; absILD1/2/4/8/16',
    'prepared_repeat_filter_reapplied': False, 'genotype_order': GENOTYPES,
    'selected_trials': len(selected), 'animals': selected.animal.nunique(),
    'genotype_counts': summary.to_dict(orient='records'),
    'bin_ms': BIN_MS, 'duration_bin_range_s': [0, float(edges[-1])],
    'no_invalid_recorded_durations': True, 'all_duration_values_and_source_lines_independently_checked': True,
    'all_histogram_counts_and_densities_independently_checked': True,
    'all_unit_areas': True, 'pooling': 'Raw trials pooled across animals, sessions and selected stimulus conditions',
    'input_unchanged': True}
(OUTPUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(summary.to_string(index=False))
print('Passed independent values, trial identities, histogram counts, unit areas and unchanged input hash.')
