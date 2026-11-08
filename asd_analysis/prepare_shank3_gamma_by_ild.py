# %% Exact existing trials and frozen parameters; no preparation filters reapplied
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from shank3_gamma_by_ild_utils import (
    ROOT, OUT, PREVIOUS, VARIANTS, EXPECTED, ABLS, ILDS, PRIORS, sha256,
    initial_parameters, parameter_names)

INPUT_NAMES = ['parent_trials.csv.gz', 'fit_trials.csv.gz', 'conditions.csv', 'membership.csv',
               'fixed_proactive.json', 'likelihood_data.npz', 'rtd_data_bins.csv',
               'fixed_parent_design_averages.npz', 'fixed_parent_design_manifest.json']
OUT.mkdir(exist_ok=True)
assert json.loads((PREVIOUS / 'completion.json').read_text())['status'] == 'complete'
baseline_path = OUT / 'preservation_baseline.json'
if not baseline_path.exists():
    baseline = json.loads((PREVIOUS / 'preservation_baseline.json').read_text())
    for path in PREVIOUS.rglob('*'):
        if path.is_file() and path.suffix in ['.csv', '.gz', '.npz', '.npy', '.pkl', '.png', '.pdf', '.py']:
            baseline[str(path.relative_to(ROOT))] = sha256(path)
    baseline_path.write_text(json.dumps(baseline, indent=2) + '\n')
baseline = json.loads(baseline_path.read_text())
for name, expected in baseline.items():
    assert sha256(ROOT / name) == expected, name
copied, runs, mapping = {}, [], []
for group in EXPECTED:
    source, dest = PREVIOUS / group, OUT / group
    dest.mkdir(exist_ok=True)
    for name in INPUT_NAMES:
        if (dest / name).exists():
            assert sha256(dest / name) == sha256(source / name), name
        else:
            shutil.copy2(source / name, dest / name)
        copied[str((dest / name).relative_to(OUT))] = dict(
            source=str((source / name).relative_to(ROOT)), sha256=sha256(source / name))
    conditions = pd.read_csv(dest / 'conditions.csv', float_precision='round_trip')
    initial = initial_parameters(conditions, 'gamma_by_ild')
    for row in conditions.itertuples():
        mapping.append(dict(group=group, condition=row.condition, ABL=row.ABL, ILD=row.ILD,
            gamma_index=ILDS.index(row.ILD), omega_index=ABLS.index(row.ABL), delay_index=0,
            valid_trials=row.trials, right=row.right, left=row.left))
    for variant in VARIANTS:
        folder = dest / variant
        folder.mkdir(exist_ok=True)
        names = parameter_names(variant)
        assert len(names) == 16
        config = dict(group=group, variant=variant, seed=0, parameter_count=16, parameter_order=names,
            initial={k:np.asarray(v).tolist() for k,v in initial.items()}, priors=PRIORS,
            learning_rate=.0002, clip_norm=1., initial_guide_scale=.1, float64=True, guide='full_rank',
            window=1000, min_steps=50000, max_steps=1000000, confirmation_windows=25,
            parameter_stability_required=False, posterior_draws=10000, diagnostic_draws=2048,
            fit_window_s=[0,1], expected_population=EXPECTED[group], initializations=1,
            input_hashes={name:sha256(dest / name) for name in INPUT_NAMES},
            initialization='Fresh seed 0; pooled signed-ILD logit Gamma; count-weighted data-derived ABL Omega')
        path = folder / 'prepared_configuration.json'
        if path.exists():
            assert json.loads(path.read_text()) == config
        else:
            path.write_text(json.dumps(config, indent=2) + '\n')
        runs.append(dict(group=group, variant=variant, parameters=16, status='prepared_not_started'))
pd.DataFrame(mapping).to_csv(OUT / 'condition_mapping.csv', index=False)
manifest = dict(status='prepared_pending_independent_verification', copied_inputs=copied, runs=runs,
    trial_filter_reapplied=False, original_source_hashes_unchanged=True,
    source=str(Path(__file__).relative_to(ROOT)), source_sha256=sha256(Path(__file__)))
(OUT / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Prepared WT and HOM: 16 free values each; exact original trials and frozen parameters.', flush=True)
