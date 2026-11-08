# %% Editable reference selection; preserve exact trial identities and old fits
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from shank3_gamma_omega_gaussian_abl_delay_utils import OUT, ROOT, REFERENCES, COUNTS, PRIORS, initialization, sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genotype', choices=['wt','hom','all'], default='all')
args=parser.parse_args()
OUT.mkdir(exist_ok=True)
SOURCE_NAMES=['fit_trials.csv.gz','parent_trials.csv.gz','conditions.csv','membership.csv',
              'likelihood_data.npz','fixed_proactive.json','excluded_valid_trials.csv',
              'fixed_parent_design_averages.npz','fixed_parent_design_manifest.json']
for group in (['wt','hom'] if args.genotype=='all' else [args.genotype]):
    ref=REFERENCES[group]
    folder=OUT/group
    folder.mkdir(exist_ok=True)
    assert not (folder/'preparation.json').exists(), 'Prepared inputs already exist; resume them.'
    prior=json.loads((ref/'preparation.json').read_text())
    assert sha256(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv')==prior['preservation_hashes']['asd_analysis/session_type_csvs/ASD_session_type_1.csv']
    for name in SOURCE_NAMES:
        shutil.copy2(ref/name,folder/name)
    mapping=pd.read_csv(folder/'conditions.csv')[['condition','ABL','ILD']]
    mapping['delay_abl_index']=mapping.ABL.map({20:0,40:1,60:2})
    mapping['delay_mu_parameter']=mapping.ABL.map(lambda a:f'delay_mu_ABL{a}')
    mapping['delay_sigma_parameter']=mapping.ABL.map(lambda a:f'delay_sigma_ABL{a}')
    mapping.to_csv(folder/'delay_condition_map.csv',index=False)
    valid=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    parent=pd.read_csv(folder/'parent_trials.csv.gz',float_precision='round_trip')
    n=COUNTS[group]
    assert len(valid)==n['valid'] and valid.animal.nunique()==n['animals']
    assert len(parent)==n['parent'] and parent.display_abort.sum()==n['aborts']
    assert valid.response_poke.value_counts().to_dict()=={1.:n['right'],-1.:n['left']}
    initial,rng=initialization()
    (folder/'initialization.json').write_text(json.dumps({k:np.asarray(v).tolist() for k,v in initial.items()},indent=2)+'\n')
    (folder/'initialization_rng.json').write_text(json.dumps(rng,indent=2)+'\n')
    preserved={str(p.relative_to(ROOT)):sha256(p) for p in ref.rglob('*') if p.is_file()}
    preserved.update(prior['preservation_hashes'])
    record=dict(genotype=group,**n,conditions=30,parameter_count=68,priors=PRIORS,
        reference=str(ref.relative_to(ROOT)),fit_window_s=[0,1],repeat_filter_reapplied=False,
        likelihood='Integrate zero-truncated Gaussian delay, then normalize complete rat race plus fixed exponential lapse per onset.',
        timing='Valid absolute RT=intended_fix_time/1000 + recorded timed_rt; abort display=fix_time/1000 - intended_fix_time/1000.',
        input_hashes={n:sha256(folder/n) for n in SOURCE_NAMES+['initialization.json','initialization_rng.json','delay_condition_map.csv']},
        preservation_hashes=preserved)
    (folder/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(group, 'prepared', n, flush=True)
