# %% Exact input copies and fresh initialization for the four authorized runs
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from shank3_tied_omega_utils import ROOT, OUT, VARIANTS, EXPECTED, PRIORS, sha256, parameter_names, initial_parameters

SOURCES = {g: ROOT / f'asd_analysis/shank3_cohort1_{g}_gamma_omega' for g in EXPECTED}
INPUT_NAMES = ['parent_trials.csv.gz', 'fit_trials.csv.gz', 'conditions.csv', 'membership.csv',
               'fixed_proactive.json', 'likelihood_data.npz', 'rtd_data_bins.csv',
               'fixed_parent_design_averages.npz', 'fixed_parent_design_manifest.json']
OUT.mkdir(exist_ok=True)
baseline_path = OUT / 'preservation_baseline.json'
if not baseline_path.exists():
    baseline = json.loads((ROOT / 'asd_analysis/shank3_cohort1_hom_gamma_omega/preparation.json').read_text())['preservation_hashes']
    for folder in [*SOURCES.values(), ROOT / 'asd_analysis/shank3_cohort1_wt_hom_comparison']:
        baseline.update({str(p.relative_to(ROOT)): sha256(p) for p in folder.rglob('*') if p.is_file()})
    baseline_path.write_text(json.dumps(baseline, indent=2) + '\n')
baseline = json.loads(baseline_path.read_text())
for name, expected in baseline.items():
    assert sha256(ROOT / name) == expected, name
copied, runs = {}, []
for group, source in SOURCES.items():
    assert json.loads((source / 'independent_validation.json').read_text())['status'] == 'passed'
    dest = OUT / group
    dest.mkdir(exist_ok=True)
    for name in INPUT_NAMES:
        if (dest / name).exists():
            assert sha256(dest / name) == sha256(source / name), name
        else:
            shutil.copy2(source / name, dest / name)
        copied[str((dest / name).relative_to(OUT))] = dict(source=str((source / name).relative_to(ROOT)), sha256=sha256(source / name))
    conditions = pd.read_csv(dest / 'conditions.csv', float_precision='round_trip')
    for variant in VARIANTS:
        folder = dest / variant
        folder.mkdir(exist_ok=True)
        initial = initial_parameters(conditions, variant)
        names = parameter_names(variant)
        assert len(names) == (36 if variant == 'omega_by_abl' else 34)
        config = dict(group=group, variant=variant, seed=0, parameter_count=len(names), parameter_order=names,
            initial={k:np.asarray(v).tolist() for k,v in initial.items()}, priors=PRIORS,
            learning_rate=.0002, clip_norm=1., initial_guide_scale=.1, float64=True, guide='full_rank',
            window=1000, min_steps=50000, max_steps=1000000, confirmation_windows=25,
            parameter_stability_required=False, posterior_draws=10000, diagnostic_draws=2048,
            fit_window_s=[0,1], expected_population=EXPECTED[group], initializations=1,
            input_hashes={name:sha256(dest / name) for name in INPUT_NAMES},
            initialization='Fresh seed 0; count-weighted data-derived condition Omega initials, not posterior means')
        path = folder / 'prepared_configuration.json'
        if path.exists():
            assert json.loads(path.read_text()) == config
        else:
            path.write_text(json.dumps(config, indent=2) + '\n')
        runs.append(dict(group=group, variant=variant, parameters=len(names), status='prepared_not_started'))
manifest = dict(status='prepared_pending_independent_verification', copied_inputs=copied, runs=runs,
                trial_filter_reapplied=False, original_source_hashes_unchanged=True,
                source=str(Path(__file__).relative_to(ROOT)), source_sha256=sha256(Path(__file__)))
(OUT / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Prepared four isolated runs (36/34 parameters per genotype); original inputs unchanged.', flush=True)
