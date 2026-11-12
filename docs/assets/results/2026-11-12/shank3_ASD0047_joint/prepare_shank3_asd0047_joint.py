# %% Reconstruct the ASD0047 animal from the current cleaned prepared CSV
from pathlib import Path
import json
import numpy as np
import pandas as pd
from shank3_asd0047_joint_utils import ROOT, OUT, OLD, ANIMALS, ABLS, ILDS, PRIORS, sha256, initialization

# %% Editable input; neither raw CSVs nor previous fitting outputs are modified
SOURCE = ROOT / 'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
COLUMNS = ['animal', 'trial', 'trial_start', 'session', 'session_type', 'training_level',
    'ABL', 'ILD', 'fix_time', 'intended_fix_time', 'timed_rt', 'response_poke', 'success',
    'abort_type', 'filename', 'genotype', 'is_short_sound', 'source_file', 'source_date']
OUT.mkdir(exist_ok=True)
assert not (OUT / 'preparation.json').exists(), 'Preserve existing prepared joint-fit inputs.'
source_hash = sha256(SOURCE)
df = pd.read_csv(SOURCE, usecols=COLUMNS, float_precision='round_trip')
df['prepared_csv_line'] = np.arange(len(df))+2
x = df.loc[df.filename.eq('SHANK3_cohort1.csv') & df.animal.isin(ANIMALS)].copy()
assert x.session_type.eq(1).all() and x.training_level.eq(16).all()
assert not x.is_short_sound.fillna(False).astype(bool).any()
x['ABL_mapped'] = x.ABL.replace({59: 60})
x = x.loc[x.ABL_mapped.isin(ABLS) & x.ILD.isin(ILDS)].copy()
assert len(x) == 21947 and set(x.animal) == set(ANIMALS)
x['onset_s'] = x.intended_fix_time/1000
x['abort_rt_s'] = x.fix_time/1000-x.onset_s
x['condition'] = x.ABL_mapped.map({a: i*10 for i, a in enumerate(ABLS)})+x.ILD.map({v: i for i, v in enumerate(ILDS)})
valid = x.success.isin([1, -1])
abort = x.abort_type.eq('Fixation') & (x.fix_time < x.intended_fix_time)
late = x.abort_type.eq('Fixation') & ~abort
fitted = valid & x.timed_rt.between(0, 1)
long = valid & (x.timed_rt > 1)
assert (valid | abort | late).all() and (fitted | long).sum() == valid.sum()
assert [int(fitted.sum()), int(abort.sum()), int(late.sum()), int(long.sum())] == [18771, 3004, 11, 161]
assert x.loc[fitted, 'response_poke'].isin([-1, 1]).all()
assert x.loc[fitted, 'response_poke'].eq(1).sum() == 9592
x['observation'] = np.select([abort, fitted, long, late], ['pre_onset_abort', 'valid_RT_0_1', 'valid_RT_above_1', 'late_fixation_censored'])
x['fit_valid'] = fitted
x['display_abort'] = abort & x.abort_rt_s.between(-1, 0, inclusive='left')
assert x.display_abort.sum() == 2923
assert (x.loc[abort, 'fix_time'] >= 0).all()
x.to_csv(OUT / 'all_eligible_parent_trials_before_RT_exclusion.csv.gz', index=False, float_format='%.17g')
x.loc[long].to_csv(OUT / 'excluded_valid_RT_above_1.csv.gz', index=False, float_format='%.17g')
x.loc[~long].to_csv(OUT / 'parent_trials.csv.gz', index=False, float_format='%.17g')
x.loc[fitted].to_csv(OUT / 'fit_trials.csv.gz', index=False, float_format='%.17g')
x.loc[abort | late].to_csv(OUT / 'abort_and_censor_trials.csv.gz', index=False, float_format='%.17g')

# %% The common contaminant support covers every retained observation window
cutoff = float(x.onset_s.max()+1.)
assert np.isclose(cutoff, 3.3)
times, counts = np.unique(x.loc[abort, 'fix_time'].to_numpy()/1000, return_counts=True)
np.savez_compressed(OUT / 'observations.npz',
    rt=x.loc[fitted, 'timed_rt'].to_numpy(), onset=x.loc[fitted, 'onset_s'].to_numpy(),
    choice=x.loc[fitted, 'response_poke'].to_numpy(int), condition=x.loc[fitted, 'condition'].to_numpy(int),
    abort_times=times, abort_counts=counts,
    late_onsets=x.loc[late, 'onset_s'].to_numpy(),
    late_condition=x.loc[late, 'condition'].to_numpy(int),
    parent_onsets=x.loc[~long, 'onset_s'].to_numpy(),
    parent_condition=x.loc[~long, 'condition'].to_numpy(int),
    cutoff=np.array(cutoff), parent_count=np.array(int((~long).sum())))
conditions = []
for i, (abl, ild) in enumerate((a, d) for a in ABLS for d in ILDS):
    z = x.loc[x.condition.eq(i)]
    v = z.loc[z.fit_valid]
    conditions.append(dict(condition=i, ABL=abl, ILD=ild, parent_trials=int(z.observation.ne('valid_RT_above_1').sum()), original_parent_trials=len(z),
        valid_trials=len(v), right=int(v.response_poke.eq(1).sum()), left=int(v.response_poke.eq(-1).sum()),
        pre_onset_aborts=int(z.observation.eq('pre_onset_abort').sum()),
        late_fixation_censored=int(z.observation.eq('late_fixation_censored').sum()),
        long_RT_excluded=int(z.observation.eq('valid_RT_above_1').sum())))
pd.DataFrame(conditions).to_csv(OUT / 'conditions.csv', index=False)
x.groupby(['animal', 'observation']).size().rename('trials').reset_index().to_csv(OUT / 'memberships.csv', index=False)

# %% Exact reference identities; prepared repeat/calibration/short-sound rules stay upstream
reference = OLD
for name, frame in [('parent_trials.csv.gz', x), ('fit_trials.csv.gz', x.loc[fitted])]:
    old = pd.read_csv(reference / name, float_precision='round_trip')
    np.testing.assert_array_equal(np.sort(frame.prepared_csv_line), np.sort(old.prepared_csv_line))
assert sha256(SOURCE) == source_hash
round_folder = OUT / 'round_00'
round_folder.mkdir()
(round_folder / 'bounds.json').write_text(json.dumps(PRIORS, indent=2)+'\n')
(round_folder / 'initialization.json').write_text(json.dumps({str(seed): {k: v.tolist() for k, v in initialization(seed).items()} for seed in [0, 1]}, indent=2)+'\n')
files = ['parent_trials.csv.gz', 'all_eligible_parent_trials_before_RT_exclusion.csv.gz', 'excluded_valid_RT_above_1.csv.gz', 'fit_trials.csv.gz', 'abort_and_censor_trials.csv.gz', 'observations.npz', 'conditions.csv', 'memberships.csv']
manifest = dict(status='prepared_confirmed_tail_exclusion_and_entry_conditioning', source=str(SOURCE), source_sha256=source_hash,
    parent=21786, original_parent=21947, valid=18771, right=9592, left=9179, abort_events=3004,
    late_fixation_censored=11, long_RT_excluded=161, display_abort=2923, display_rows=21694,
    animals=ANIMALS, conditions=30, free_parameters=20, contaminant_cutoff_s=cutoff,
    contaminant_cutoff_rule='Common fixation-time cutoff=max planned onset+1s; fixed from the trial design, not fitted.',
    repeat_filter_reapplied=False, exact_reference_trial_identities=True,
    input_hashes={name: sha256(OUT / name) for name in files})
(OUT / 'preparation.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2), flush=True)
