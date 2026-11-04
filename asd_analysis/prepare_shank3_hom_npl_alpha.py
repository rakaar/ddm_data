# %% Editable source roots and preparation of immutable fit inputs
"""Copy the exact verified HOM trial population; independently verify before fitting."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from shank3_hom_npl_alpha_utils import ROOT, OUT, REFERENCE, PRIORS, INITIAL, parameter_names, sha256

SOURCE_NAMES = ['fit_trials.csv.gz', 'parent_trials.csv.gz', 'conditions.csv', 'membership.csv',
                'likelihood_data.npz', 'fixed_proactive.json', 'excluded_valid_trials.csv']
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'preparation.json').exists(), 'Existing preparation is preserved; resume the fit instead.'
reference_prep = json.loads((REFERENCE/'preparation.json').read_text())
csv = ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
assert sha256(csv) == reference_prep['preservation_hashes'][str(csv.relative_to(ROOT))]
for name in SOURCE_NAMES:
    shutil.copy2(REFERENCE/name, OUT/name)
trials = pd.read_csv(OUT/'fit_trials.csv.gz', float_precision='round_trip')
assert len(trials) == 67715 and trials.animal.nunique() == 6
assert trials.response_poke.value_counts().to_dict() == {1.: 35320, -1.: 32395}
assert trials.timed_rt.between(0, 1).all()
assert len(parameter_names()) == 37

# Freeze hashes of the inputs and prior fit outputs, including the source CSVs.
preserve = [csv, ROOT/'SHANK3_cohort1.csv', ROOT/'asd_analysis/session_type_csvs/ASD_session_type_2.csv',
            ROOT/'asd_analysis/session_type_csvs/ASD_session_type_23.csv']
for folder in [REFERENCE, ROOT/'asd_analysis/shank3_cohort1_proactive_lapse',
               ROOT/'asd_analysis/shank3_cohort1_wt_npl_alpha_condition_delay']:
    preserve.extend(p for p in folder.rglob('*') if p.is_file())
preserve.extend(ROOT/'asd_analysis'/name for name in [
    'shank3_wt_npl_alpha_utils.py', 'prepare_shank3_wt_npl_alpha.py', 'fit_shank3_wt_npl_alpha.py',
    'plot_shank3_wt_npl_alpha.py', 'verify_shank3_wt_npl_alpha.py',
    'publish_shank3_wt_npl_alpha.py', 'run_shank3_wt_npl_alpha_pipeline.py'])
record = dict(valid_trials=67715, right=35320, left=32395, animals=6, conditions=30,
    parent_trials=81721, display_aborts=13274, display_trials=80989, excluded_valid_above_1=357,
    repeat_filter_reapplied=False, parameter_count=37, fit_window_s=[0, 1],
    priors=PRIORS, initial={k: np.asarray(v).tolist() for k,v in INITIAL.items()},
    reference=str(REFERENCE.relative_to(ROOT)),
    input_hashes={name: sha256(OUT/name) for name in SOURCE_NAMES},
    preservation_hashes={str(p.relative_to(ROOT)): sha256(p) for p in preserve},
    timing='Valid: onset=intended_fix_time/1000, absolute response=onset+recorded timed_rt. Abort display: fix_time/1000-onset.',
    likelihood='Complete proactive/reactive race plus fixed exponential lapse; conditional RT+choice on inclusive [0,1].',
    uncertainty='Conditional on fixed HOM seed-0 proactive/lapse physical posterior means; one initialization.')
(OUT/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
print('Prepared 67,715 choices, 6 animals, 30 conditions; 37 free parameters:', OUT, flush=True)
